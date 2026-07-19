"""
Local addition -- no upstream equivalent.

MTP (multi-token-prediction / speculative-decoding head) support for expert
pruning. Some checkpoints of this architecture ship an extra `mtp.*` decoder layer
plus a small "predictor" module (a norm + linear combining the model's final
hidden state with the actual next-token embedding). It shares the main model's
global expert count (`config.num_experts`) but has an INDEPENDENTLY LEARNED router
and expert pool -- pruning it correctly means giving it its own saliency pass, not
reusing the main model's retained-expert indices or leaving it untouched with a
stale expert count.

Not every checkpoint of this architecture ships this head. Every public function
here is a no-op or returns False when `mtp.*` tensors aren't present in the
checkpoint's index, so callers (el_prune.py / el_prune_disk_surgery.py) can call
this unconditionally without needing to know ahead of time whether a given
checkpoint has MTP.

Design note: rather than hand-implement MTP's self-attention (RoPE, GQA, causal
masking), this builds a REAL `Qwen3_5MoeDecoderLayer` from `transformers` (the same
class the main model's own full_attention blocks use) and feeds it the exact
attention_mask/position_ids/position_embeddings the model's own last full_attention
block actually received (captured during the main calibration run -- see
layerwise_observer.py's final_hidden_states/final_block_kwargs and
layerwise_prune.py's mtp_replay_*.pt output). This is valid because MTP's layer has
the same layer_type ("full_attention") and config as that block, and per-position
RoPE/mask metadata doesn't depend on which specific full_attention layer receives
it. The only genuinely new math here is the small "predictor" combination step
(pre_fc_norm_hidden/pre_fc_norm_embedding/fc) and the unfused-per-expert-file ->
fused-3D-tensor conversion for MTP's own expert pool.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

from reap.disk_stream_util import SafetensorsIndex
from reap.layerwise_observer import LayerwiseMoEObserver

logger = logging.getLogger(__name__)

MTP_PREFIX = "mtp."
MTP_LAYER_PREFIX = "mtp.layers.0."


def has_mtp(disk_index: SafetensorsIndex) -> bool:
    return any(name.startswith(MTP_PREFIX) for name in disk_index.weight_map)


def _extract_input_ids(batch: Any) -> torch.Tensor:
    """Mirrors layerwise_observer.py's _capture_first_block_inputs input handling
    -- data_batches elements are either a raw input_ids tensor or a dict/BatchEncoding
    with an "input_ids" key."""
    if isinstance(batch, torch.Tensor):
        return batch.unsqueeze(0) if batch.dim() == 1 else batch
    if hasattr(batch, "items"):
        tensor = batch["input_ids"]
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
    raise ValueError(f"Unsupported calibration batch type: {type(batch)}")


def build_mtp_layer(
    text_config, disk_index: SafetensorsIndex, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    """Build a real Qwen3_5MoeDecoderLayer with MTP's actual weights loaded.

    Instantiated at layer_idx = last index of config.layer_types, since MTP's layer
    is documented (vLLM's Qwen3_5MultiTokenPredictor) to be a full_attention layer
    like the model's own final block -- this gives the right submodule shapes
    (self_attn, not linear_attn) for free rather than hardcoding attention dims.
    """
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_modeling

    last_layer_idx = len(text_config.layer_types) - 1
    if text_config.layer_types[last_layer_idx] != "full_attention":
        raise ValueError(
            f"Expected the model's last layer_type to be 'full_attention' (MTP's "
            f"documented layer type) but got '{text_config.layer_types[last_layer_idx]}' "
            f"-- MTP layer construction assumptions may not hold for this checkpoint."
        )

    layer = qwen3_5_modeling.Qwen3_5MoeDecoderLayer(text_config, last_layer_idx)
    layer = layer.to(device=device, dtype=dtype)

    def load(name: str) -> torch.Tensor:
        return disk_index.read_tensor(name).to(device=device, dtype=dtype)

    state_dict: Dict[str, torch.Tensor] = {}
    direct_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    ]
    for rel_name in direct_names:
        full_name = MTP_LAYER_PREFIX + rel_name
        if full_name in disk_index.weight_map:
            state_dict[rel_name] = load(full_name)
        else:
            logger.warning(f"MTP tensor {full_name} not in checkpoint; leaving {rel_name} at init value")

    num_experts = text_config.num_experts
    gate_up_rows = []
    down_rows = []
    for expert_idx in range(num_experts):
        gate_w = load(f"{MTP_LAYER_PREFIX}mlp.experts.{expert_idx}.gate_proj.weight")
        up_w = load(f"{MTP_LAYER_PREFIX}mlp.experts.{expert_idx}.up_proj.weight")
        down_w = load(f"{MTP_LAYER_PREFIX}mlp.experts.{expert_idx}.down_proj.weight")
        gate_up_rows.append(torch.cat([gate_w, up_w], dim=0))
        down_rows.append(down_w)
    state_dict["mlp.experts.gate_up_proj"] = torch.stack(gate_up_rows, dim=0)
    state_dict["mlp.experts.down_proj"] = torch.stack(down_rows, dim=0)

    missing, unexpected = layer.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys building MTP layer: {unexpected}")
    if missing:
        logger.warning(f"MTP layer left un-loaded (init/random) keys: {missing}")

    return layer


def build_predictor_weights(
    disk_index: SafetensorsIndex, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> Dict[str, torch.Tensor]:
    def load(name: str) -> torch.Tensor:
        return disk_index.read_tensor(name).to(device=device, dtype=dtype)

    return {
        "fc_weight": load("mtp.fc.weight"),
        "pre_fc_norm_embedding_weight": load("mtp.pre_fc_norm_embedding.weight"),
        "pre_fc_norm_hidden_weight": load("mtp.pre_fc_norm_hidden.weight"),
    }


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Matches Qwen3_5MoeRMSNorm's forward exactly (float32 accumulation)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


class _MTPModelWrapper(nn.Module):
    """Minimal fake "model" whose module tree matches what
    reap.layerwise_model_utils.extract_model_components looks for (a
    `.model.language_model.layers` ModuleList of decoder blocks), so the real
    LayerwiseMoEObserver machinery -- hooks, per-expert stat accumulation, block
    replay -- can be reused unmodified for MTP's single layer instead of
    duplicating that logic."""

    def __init__(self, mtp_layer: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([mtp_layer])


def run_mtp_observer(
    text_config,
    disk_index: SafetensorsIndex,
    mtp_replay_data: Dict[str, Any],
    hook_config,
    embed_tokens_weight: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[int, Dict[str, Any]]:
    """Run MTP's own saliency pass and return observer-data in the same
    {0: {"expert_frequency": ..., "reap": ..., ...}} shape reap.prune's
    select_retained_experts expects (single "layer" 0 = MTP's one layer)."""
    final_hidden_states: List[torch.Tensor] = mtp_replay_data["final_hidden_states"]
    final_block_kwargs: List[Dict[str, Any]] = mtp_replay_data["final_block_kwargs"]
    input_ids_batches: List[Any] = mtp_replay_data["input_ids_batches"]

    if not (len(final_hidden_states) == len(final_block_kwargs) == len(input_ids_batches)):
        raise ValueError(
            f"mtp_replay_data batch counts disagree: "
            f"{len(final_hidden_states)} hidden states, {len(final_block_kwargs)} kwargs, "
            f"{len(input_ids_batches)} input_ids batches"
        )

    # REAP's own replay machinery runs the main model's calibration forward in
    # float32 regardless of the checkpoint's storage dtype (RoPE cos/sin come back
    # float32 and promote hidden_states through the chain) -- build the MTP layer
    # and predictor weights in whatever dtype the captured replay data actually
    # used, not the checkpoint's on-disk dtype, or every op mixing the two raises a
    # dtype mismatch the moment RoPE or the attention mask touches a tensor.
    replay_dtype = final_hidden_states[0].dtype

    mtp_layer = build_mtp_layer(text_config, disk_index, device=device, dtype=replay_dtype)
    predictor = build_predictor_weights(disk_index, device=device, dtype=replay_dtype)
    embed_tokens_weight = embed_tokens_weight.to(device=device, dtype=replay_dtype)

    fake_model = _MTPModelWrapper(mtp_layer)
    observer = LayerwiseMoEObserver(model=fake_model, hook_config=hook_config, disk_index=None)

    for hidden_states, block_kwargs, raw_batch in zip(
        final_hidden_states, final_block_kwargs, input_ids_batches
    ):
        hidden_states = hidden_states.to(device=device, dtype=replay_dtype)
        input_ids = _extract_input_ids(raw_batch).to(device)

        # Teacher-forcing target: the actual next token at each position. There's
        # no "next token" for the last position -- rather than slice every kwarg
        # tensor (masks/position_ids/position_embeddings) down to seq_len-1, which
        # is easy to get subtly wrong for arbitrary mask shapes, shift left and pad
        # with a repeat of the last real token. This makes the final position's
        # target technically wrong, but it's one out of a full sequence length of
        # tokens aggregated across many calibration batches -- negligible for
        # saliency statistics, and keeps every tensor's sequence length identical
        # so the captured kwargs can be reused completely unmodified.
        next_token_ids = torch.cat([input_ids[:, 1:], input_ids[:, -1:]], dim=1)
        next_token_embed = nn.functional.embedding(next_token_ids, embed_tokens_weight)

        norm_hidden = _rms_norm(
            hidden_states, predictor["pre_fc_norm_hidden_weight"], text_config.rms_norm_eps
        )
        norm_embed = _rms_norm(
            next_token_embed, predictor["pre_fc_norm_embedding_weight"], text_config.rms_norm_eps
        )
        fc_input = torch.cat([norm_embed, norm_hidden], dim=-1)
        mtp_input_hidden_states = nn.functional.linear(fc_input, predictor["fc_weight"])

        seed_kwargs = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in block_kwargs.items()
        }
        observer.replay_cache.append(inputs=[mtp_input_hidden_states], kwargs=seed_kwargs)

    observer._record_activations_for_block(0, moe_module=mtp_layer.mlp)
    observer.replay_cache.clear()

    # report_state() finalizes any OnlineStatsTracker accumulators into plain
    # tensors -- select_retained_experts (like the main model's own path) expects
    # already-finalized saliency tensors, not the raw in-progress trackers.
    return observer.report_state()
