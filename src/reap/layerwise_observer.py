"""
Layerwise MoE Observer for memory-efficient expert pruning calibration.

This module implements a block-wise activation collection approach inspired by AutoGPTQ,
adapted for MoE expert pruning metrics (REAP, EAN, frequency, etc.).

Key features:
1. Only one transformer block is loaded on GPU at a time
2. Hidden states are cached between blocks (passed from block N to block N+1)
3. Streaming approach - batches are processed one at a time
4. Progressive loading/offloading of transformer blocks
5. Computes all REAP pruning metrics per block
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import gc
import inspect
import logging
import pathlib

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from reap.observer import (
    MoETransformerObserverConfig,
)
from reap.layerwise_model_utils import (
    extract_model_components,
    get_module_by_name,
    find_decoder_blocks,
    cleanup_memory,
    move_to_device,
    safe_get_device,
    has_meta_tensors,
)
from reap.pruning_metrics import initialize_pruning_state, update_pruning_state
from reap.metrics import OnlineStatsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _FirstblockInputCaptured(Exception):
    """Internal sentinel used to stop execution after caching block-0 inputs."""


@dataclass
class ReplayBatch:
    """Cached replay payload for one calibration batch."""

    inputs: List[torch.Tensor]
    kwargs: Dict[str, Any]
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


class ReplayCache:
    """Manage cached replay payloads between layerwise block forwards."""

    def __init__(self) -> None:
        self._batches: List[ReplayBatch] = []

    def __len__(self) -> int:
        return len(self._batches)

    def append(
        self,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self._batches.append(
            ReplayBatch(
                inputs=inputs,
                kwargs=kwargs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        )

    def clear(self) -> None:
        self._batches.clear()

    def materialize(
        self,
        batch_idx: int,
        target_device: torch.device,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        batch = self._batches[batch_idx]

        replay_inputs = [
            tensor if str(tensor.device) == "meta" else tensor.to(target_device)
            for tensor in batch.inputs
        ]

        replay_kwargs = {
            key: move_to_device(value, target_device)
            for key, value in batch.kwargs.items()
        }

        if batch.attention_mask is not None:
            replay_kwargs["attention_mask"] = move_to_device(
                batch.attention_mask, target_device
            )
        if batch.position_ids is not None:
            replay_kwargs["position_ids"] = move_to_device(
                batch.position_ids, target_device
            )

        return replay_inputs, replay_kwargs

    def replace_inputs(self, next_inputs: List[List[torch.Tensor]]) -> None:
        if len(next_inputs) != len(self._batches):
            raise ValueError(
                "Replacement inputs must match the replay cache batch count"
            )

        for batch, inputs in zip(self._batches, next_inputs):
            batch.inputs = inputs


class LayerwiseMoEObserver:
    """
    Memory-efficient MoE observer that processes one transformer block at a time.

    This class collects the same pruning metrics as MoETransformerObserver but
    in a memory-efficient manner suitable for large models on single GPUs.

    Metrics collected per block:
    - total_tokens: Total number of tokens processed
    - expert_frequency: How often each expert is selected
    - pairwise_expert_frequency: Co-occurrence counts
    - ean_sum: Sum of L2 norms of expert outputs (for routed tokens)
    - ean_mean: Mean of L2 norms
    - weighted_ean_sum: Router-weighted EAN
    - reap: Mean of (router_weight * activation_norm)
    - weighted_expert_frequency_sum: Sum of router weights per expert
    - max_activations: Maximum activation magnitude per expert
    """

    _REPLAY_KWARG_DROP_KEYS = {"past_key_value", "past_key_values"}
    _REPLAY_KWARG_FORCED_VALUES = {
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": False,
        "output_router_loss": False,
        "output_gate_logits": False,
    }

    def __init__(
        self,
        model: nn.Module,
        hook_config: MoETransformerObserverConfig,
        block_names: Optional[List[str]] = None,
    ):
        """
        Initialize the layerwise (blockwise) MoE observer.

        Args:
            model: The PyTorch MoE model to observe
            hook_config: Configuration for hooks (contains MoE-specific settings)
            block_names: List of transformer block names. Auto-detected if None.
        """
        self.model = model
        self.hook_config = hook_config
        self._memory_cleanup_freq = 4

        # Auto-detect decoder blocks if not provided
        self.block_names = block_names or find_decoder_blocks(self.model)

        # Extract model components
        self.blocks, self.non_backbone_modules = extract_model_components(
            self.model, self.block_names
        )

        # Cache for replaying block inputs and forward kwargs between blocks
        self.replay_cache = ReplayCache()

        # Track which block is currently loaded
        self.currently_loaded_block_idx = -1

        # Hooks for current block
        self.hooks = []

        # State dictionary to store metrics per block
        self.state: Dict[int, Dict[str, Any]] = {}

        # MoE module cache per block
        self._moe_modules_cache: Dict[int, Optional[nn.Module]] = {}

        # Forward signature cache per block
        self._forward_signature_cache: Dict[int, Tuple[set[str], bool]] = {}

        logger.info(
            f"LayerwiseMoEObserver initialized with {len(self.block_names)} blocks"
        )
        logger.info(
            f"Block names: {self.block_names[:3]}{'...' if len(self.block_names) > 3 else ''}"
        )

    def _find_moe_module_in_block(self, block_idx: int) -> Optional[nn.Module]:
        """Find the MoE module within a transformer block."""
        if block_idx in self._moe_modules_cache:
            return self._moe_modules_cache[block_idx]

        if not self.blocks or block_idx >= len(self.blocks):
            return None

        block = self.blocks[block_idx]
        block_name = self.block_names[block_idx]
            
        # Search for MoE module by class name pattern from hook config
        moe_class_name = self.hook_config.module_class_name_to_hook_regex

        for name, module in block.named_modules():
            if module.__class__.__name__ == moe_class_name:
                self._moe_modules_cache[block_idx] = module
                logger.debug(
                    f"Found MoE module at {block_name}.{name}: {module.__class__.__name__}"
                )
                return module

        logger.warning(
            f"No MoE module found in block {block_idx} matching {moe_class_name}"
        )
        self._moe_modules_cache[block_idx] = None
        return None

    def _initialize_block_state(self, num_experts: int) -> Dict[str, Any]:
        """Initialize state dictionary for a block."""
        return initialize_pruning_state(num_experts)

    def _block_at(self, block_idx: int):
        """Return the block for a valid non-negative block index, else None."""
        if not self.blocks or not (0 <= block_idx < len(self.blocks)):
            return None
        return self.blocks[block_idx]

    def _move_block(self, block, block_idx: int, device: str) -> str:
        """
        Move a block to the requested device when possible.
        Returns the block's final device.
        """
        try:
            if has_meta_tensors(block):
                logger.debug(
                    "Block %s has meta tensors, skipping move to %s",
                    block_idx,
                    device,
                )
                return safe_get_device(block)

            current_device = safe_get_device(block)
            if current_device != device:
                block.to(device)
                logger.debug(
                    "Moved block %s from %s to %s",
                    block_idx,
                    current_device,
                    device,
                )
        except Exception as exc:
            logger.warning(
                "Could not move block %s to %s: %s",
                block_idx,
                device,
                exc,
            )

        return safe_get_device(block)

    def _load_block_for_replay(self, block_idx: int) -> str:
        """Load the requested transformer block and unload the previous one."""
        block = self._block_at(block_idx)
        if block is None:
            raise IndexError("Invalid block index: %s", block_idx)

        if self.currently_loaded_block_idx == block_idx:
            return safe_get_device(block)

        self._offload_current_block()

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        final_device = self._move_block(block, block_idx, target_device)

        self.currently_loaded_block_idx = block_idx
        logger.debug("Loaded block %s", block_idx)
        return final_device

    def _offload_current_block(self) -> None:
        """Offload the current block to CPU and release memory."""
        block_idx = self.currently_loaded_block_idx
        if block_idx < 0:
            return

        block = self._block_at(block_idx)

        try:
            if block is not None:
                self._move_block(block, block_idx, "cpu")
        finally:
            self.currently_loaded_block_idx = -1
            cleanup_memory(synchronize=False)

    def _capture_first_block_inputs(self, data_batches: List[torch.Tensor]):
        """
        Run calibration batches just far enough to cache the tensors entering
        block 0.

        Args:
            data_batches: Calibration batches or token tensors
        """
        if not self.blocks:
            raise ValueError("Layerwise replay requires at least one transformer block")

        self.replay_cache.clear()

        entry_block = self.blocks[0]
        cpu_device = torch.device("cpu")
        captured_batches: List[ReplayBatch] = []

        def select_entrypoint_context() -> Tuple[torch.device, Optional[torch.dtype]]:
            chosen_device: Optional[torch.device] = None
            chosen_dtype: Optional[torch.dtype] = None

            def scan_module(module: nn.Module) -> None:
                nonlocal chosen_device, chosen_dtype

                for parameter in module.parameters():
                    if str(parameter.device) == "meta":
                        continue
                    if chosen_device is None:
                        chosen_device = parameter.device
                    if chosen_dtype is None:
                        chosen_dtype = parameter.dtype
                    if chosen_device is not None and chosen_dtype is not None:
                        return

                for buffer in module.buffers():
                    if str(buffer.device) == "meta":
                        continue
                    if chosen_device is None:
                        chosen_device = buffer.device
                    if chosen_dtype is None and torch.is_floating_point(buffer):
                        chosen_dtype = buffer.dtype
                    if chosen_device is not None and chosen_dtype is not None:
                        return

            for module_name in self.non_backbone_modules:
                module = get_module_by_name(self.model, module_name)
                if module is None:
                    continue
                try:
                    scan_module(module)
                except Exception:
                    continue
                if chosen_device is not None and chosen_dtype is not None:
                    break

            try:
                if chosen_dtype is None:
                    for parameter in entry_block.parameters():
                        if str(parameter.device) == "meta":
                            continue
                        if chosen_device is None:
                            chosen_device = parameter.device
                        chosen_dtype = parameter.dtype
                        break

                if chosen_device is None:
                    for buffer in entry_block.buffers():
                        if str(buffer.device) == "meta":
                            continue
                        chosen_device = buffer.device
                        if chosen_dtype is None and torch.is_floating_point(buffer):
                            chosen_dtype = buffer.dtype
                        break
            except Exception:
                pass

            if chosen_device is None:
                chosen_device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

            return chosen_device, chosen_dtype

        seed_device, model_dtype = select_entrypoint_context()

        def prepare_model_inputs(
            batch: torch.Tensor | Dict[str, Any] | BatchEncoding,
        ) -> Dict[str, Any]:
            if isinstance(batch, torch.Tensor):
                input_ids = batch.unsqueeze(0) if batch.dim() == 1 else batch
                return {"input_ids": input_ids.to(seed_device)}

            if isinstance(batch, (dict, BatchEncoding)):
                prepared: Dict[str, Any] = {}
                for key, value in batch.items():
                    if not torch.is_tensor(value):
                        prepared[key] = value
                        continue

                    tensor = value.unsqueeze(0) if value.dim() == 1 else value
                    if tensor.is_floating_point() and model_dtype is not None:
                        tensor = tensor.to(dtype=model_dtype)
                    prepared[key] = tensor.to(seed_device)
                return prepared

            raise ValueError(f"Unsupported batch type: {type(batch)}")

        entry_signature = inspect.signature(entry_block.forward)
        entry_param_names = [
            name
            for name, parameter in entry_signature.parameters.items()
            if name != "self"
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        def intercept_entry_inputs(_, args, kwargs):
            """Capture first-block inputs without assuming masks only arrive in kwargs.

            ERNIE-style decoder layers may pass `attention_mask` and `position_ids`
            positionally alongside non-tensor arguments, so we recover them by the
            entry block's parameter names and only cache `hidden_states` as replay input.
            """
            replay_kwargs = {}
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")

            replay_inputs: List[torch.Tensor] = []
            for index, value in enumerate(args):
                param_name = (
                    entry_param_names[index] if index < len(entry_param_names) else None
                )

                if index == 0 or param_name == "hidden_states":
                    if not torch.is_tensor(value):
                        raise TypeError(
                            "Expected first decoder block input hidden_states to be a tensor"
                        )
                    replay_inputs.append(value.detach().cpu())
                    continue

                if param_name == "attention_mask":
                    attention_mask = value
                    continue

                if param_name == "position_ids":
                    position_ids = value
                    continue

                if param_name is not None:
                    replay_kwargs[param_name] = move_to_device(value, cpu_device)

            for key, value in kwargs.items():
                if key in {"hidden_states", "attention_mask", "position_ids"}:
                    continue
                replay_kwargs[key] = move_to_device(value, cpu_device)

            captured_batches.append(
                ReplayBatch(
                    inputs=replay_inputs,
                    kwargs=LayerwiseMoEObserver._sanitize_cached_block_kwargs(
                        replay_kwargs
                    ),
                    attention_mask=attention_mask.detach().cpu()
                    if torch.is_tensor(attention_mask)
                    else None,
                    position_ids=position_ids.detach().cpu()
                    if torch.is_tensor(position_ids)
                    else None,
                )
            )

            raise _FirstblockInputCaptured

        logger.info("Seeding replay cache from the first decoder block")
        hook_handle = entry_block.register_forward_pre_hook(
            intercept_entry_inputs, with_kwargs=True
        )

        try:
            for batch in tqdm(data_batches, desc="Seeding replay cache"):
                try:
                    self.model(**prepare_model_inputs(batch))
                except _FirstblockInputCaptured:
                    continue
        finally:
            hook_handle.remove()

        for batch in captured_batches:
            self.replay_cache.append(
                inputs=batch.inputs,
                kwargs=batch.kwargs,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
            )

        logger.info("Prepared replay cache for %s batches", len(self.replay_cache))

        if not self.replay_cache:
            raise ValueError("Replay cache did not capture any first-block inputs")

    @classmethod
    def _sanitize_cached_block_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop cache kwargs that cannot be replayed safely."""
        sanitized = {}
        for key, value in kwargs.items():
            if key in cls._REPLAY_KWARG_DROP_KEYS:
                continue
            sanitized[key] = value
        return sanitized

    def _get_forward_signature_info(self, block_idx: int) -> Tuple[set[str], bool]:
        """Return accepted forward kwargs and whether the block accepts **kwargs."""
        cached = self._forward_signature_cache.get(block_idx)
        if cached is not None:
            return cached

        signature = inspect.signature(self.blocks[block_idx].forward)
        accepted_kwargs = set()
        accepts_var_kwargs = False
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_var_kwargs = True
            elif parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted_kwargs.add(name)

        info = (accepted_kwargs, accepts_var_kwargs)
        self._forward_signature_cache[block_idx] = info
        return info

    def _build_replay_kwargs(self, block_idx: int, block_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for replaying a decoder block without cache state."""
        replay_kwargs = self._sanitize_cached_block_kwargs(block_kwargs)
        accepted_kwargs, accepts_var_kwargs = self._get_forward_signature_info(block_idx)

        for key, value in self._REPLAY_KWARG_FORCED_VALUES.items():
            if accepts_var_kwargs or key in accepted_kwargs:
                replay_kwargs[key] = value

        if accepts_var_kwargs:
            return replay_kwargs

        return {key: value for key, value in replay_kwargs.items() if key in accepted_kwargs}    

    @torch.inference_mode()
    def _process_moe_activations(
        self,
        block_idx: int,
        moe_module: nn.Module,
        input_hidden_states: torch.Tensor,
        device: torch.device,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Process MoE activations and compute pruning metrics.

        This is the core function that computes REAP metrics for a single batch
        through a single MoE block.

        Args:
            block_idx: Index of the transformer block
            moe_module: The MoE module to process
            input_hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim]
            device: Target device for computation
            attention_mask: Optional attention mask of shape [batch_size, seq_len] or
                           [batch_size, 1, seq_len, seq_len]. If provided, padding tokens
                           (where mask is 0) are excluded from metric computation.
        """
        from functools import reduce

        # Get MoE configuration from hook config
        num_experts = reduce(
            getattr, self.hook_config.num_experts_attr_name.split("."), moe_module
        )
        top_k = reduce(getattr, self.hook_config.top_k_attr_name.split("."), moe_module)

        if num_experts is None or top_k is None:
            raise ValueError(
                f"MoE module at block {block_idx} missing num_experts or top_k attributes"
            )

        batch_size, sequence_length, hidden_dim = input_hidden_states.shape
        flat_input = input_hidden_states.view(-1, hidden_dim)

        # Create valid token mask from attention mask
        # This filters out padding tokens from metric computation
        valid_token_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            # Handle different attention mask shapes
            if attention_mask.dim() == 4:
                # Shape: [batch_size, 1, seq_len, seq_len] - HuggingFace 4D causal mask
                # Use the last row of each batch's mask to infer which token
                # positions are valid for the full sequence. Some models pass
                # a boolean mask (True = valid), others an additive mask
                # (0 = valid, large negative = masked).
                mask_row = attention_mask[:, 0, -1, :]
                if mask_row.dtype == torch.bool:
                    valid_token_mask = mask_row
                else:
                    valid_token_mask = mask_row == 0
            elif attention_mask.dim() == 2:
                # Shape: [batch_size, seq_len] - standard padding mask
                # Convention: 1 = valid, 0 = padding.
                valid_token_mask = attention_mask.bool()
            else:
                logger.warning(
                    f"Unexpected attention_mask shape {attention_mask.shape}, ignoring"
                )

            if valid_token_mask is not None:
                # Flatten to [batch_size * seq_len]
                valid_token_mask = valid_token_mask.reshape(-1)

        # Initialize state for this block if needed
        if block_idx not in self.state:
            self.state[block_idx] = self._initialize_block_state(num_experts)

        # Compute activations for all experts
        activations = torch.zeros((num_experts, *flat_input.shape), device=device)

        # TODO(ivanl): model-specific handling of router_module return signature
        def extract_router_logits(router_module, input):
            """Call routers that expect either flattened or sequence-shaped hidden states.

            DeepSeek's gate unpacks `[batch, seq, hidden]` internally, while other
            routers accept the flattened `[tokens, hidden]` view used for metric
            collection. Retry with the original 3D shape when the flat call fails.
            """
            try:
                result = router_module(input)
            except (TypeError, ValueError):
                if input.ndim != 2:
                    raise
                result = router_module(
                    input.view(batch_size, sequence_length, hidden_dim)
                )
            if isinstance(result, tuple):
                *_, router_logits = result
            else:
                router_logits = result
            return router_logits

        if self.hook_config.fused_experts:
            # Fused experts (e.g., Llama-4)
            router_logits = extract_router_logits(moe_module.router, flat_input)
            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
            selected_experts = selected_experts.to(device)

            router_indices = (
                torch.arange(batch_size * sequence_length, device=device)
                .view(1, -1)
                .expand(num_experts, -1)
            )
            router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
            routed_in = torch.gather(
                input=flat_input,
                dim=0,
                index=router_indices,
            ).to(device)
            routed_out = moe_module.experts(routed_in)
            activations = routed_out.view(num_experts, *flat_input.shape)
        else:
            # Loop-based MoE execution
            # First, we need to get router logits by doing a forward pass
            # This is done via the router in the MoE module
            if hasattr(moe_module, "gate"):
                router_logits = extract_router_logits(moe_module.gate, flat_input)
            elif hasattr(moe_module, "router"):
                router_logits = extract_router_logits(moe_module.router, flat_input)
            else:
                raise ValueError(
                    f"Cannot find router in MoE module at block {block_idx}"
                )

            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)

            # Compute activations for all experts
            for idx, expert in enumerate(moe_module.experts):
                activations[idx] = expert(flat_input).to(device)

        update_pruning_state(
            self.state[block_idx],
            activations=activations,
            selected_experts=selected_experts,
            router_logits=router_logits,
            num_experts=num_experts,
            valid_token_mask=valid_token_mask,
            renormalize_router_weights=self.hook_config.renormalize_router_weights,
        )

        # Clean up
        del activations, selected_experts, router_logits
        if valid_token_mask is not None:
            del valid_token_mask
        gc.collect()

    @torch.inference_mode()
    def _forward_block(
        self,
        block_idx: int,
        before_forward: Optional[Callable[[], None]] = None,
        after_forward: Optional[Callable[[torch.device, Optional[torch.Tensor]], None]] = None,
    ) -> Dict[str, Any]:
        """Forward cached hidden states through a single transformer block."""
        block_name = (
            self.block_names[block_idx]
            if block_idx < len(self.block_names)
            else f"block_{block_idx}"
        )
        logger.info(
            f"Processing block {block_idx + 1}/{len(self.blocks)}: {block_name}"
        )

        self.model.eval()

        if not self.blocks or block_idx >= len(self.blocks):
            raise ValueError(f"Block {block_idx} not found")

        device_str = self._load_block_for_replay(block_idx)
        block = self.blocks[block_idx]

        if device_str == "meta":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device_str)

        try:
            if block_idx == 0 and not self.replay_cache:
                raise ValueError(
                    "First block inputs have not been captured; call "
                    "_capture_first_block_inputs before forwarding block 0"
                )
            if not self.replay_cache:
                raise ValueError("No cached block inputs available")

            num_batches = len(self.replay_cache)
            block_outputs = []

            for batch_idx in tqdm(range(num_batches), desc=f"Processing {block_name}"):
                
                block_input, block_kwargs = self.replay_cache.materialize(
                    batch_idx=batch_idx,
                    target_device=target_device,
                )
                attention_mask = block_kwargs.get("attention_mask", None)

                block_kwargs = self._build_replay_kwargs(block_idx, block_kwargs)
                if before_forward is not None:
                    before_forward()

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    outputs = block(*block_input, **block_kwargs)

                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs

                if after_forward is not None:
                    after_forward(target_device, attention_mask)

                block_outputs.append([hidden_states.detach().cpu()])

                del outputs, hidden_states, block_input, block_kwargs

                if batch_idx % self._memory_cleanup_freq == 0:
                    cleanup_memory(synchronize=False)

            if block_idx < len(self.blocks) - 1:
                self.replay_cache.replace_inputs(block_outputs)

            logger.info(f"Completed block {block_idx}: processed {num_batches} batches")

            return self.state.get(block_idx, {})

        finally:
            self._offload_current_block()

    @torch.inference_mode()
    def _record_activations_for_block(
        self,
        block_idx: int,
        moe_module: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Record MoE activations and compute metrics for a single block.

        Args:
            block_idx: Index of the block to process
            moe_module: Optional pre-resolved MoE module for this block

        Returns:
            Dictionary with computed metrics for this block
        """
        if moe_module is None:
            moe_module = self._find_moe_module_in_block(block_idx)
            if moe_module is None:
                return self._forward_block(block_idx)

        captured_moe_input: Dict[str, torch.Tensor] = {}
        moe_hook_handle = None

        def _capture_moe_input_hook(module, args, output):
            captured_moe_input["input"] = args[0].detach()
            return output

        def _before_forward() -> None:
            captured_moe_input.clear()

        def _after_forward(
            target_device: torch.device,
            attention_mask: Optional[torch.Tensor],
        ) -> None:
            moe_input = captured_moe_input.get("input")
            if moe_input is None:
                raise RuntimeError(f"Failed to capture MoE input for block {block_idx}")

            self._process_moe_activations(
                block_idx,
                moe_module,
                moe_input,
                target_device,
                attention_mask=attention_mask,
            )

            del moe_input
            captured_moe_input.clear()

        moe_hook_handle = moe_module.register_forward_hook(_capture_moe_input_hook)

        try:
            return self._forward_block(
                block_idx,
                before_forward=_before_forward,
                after_forward=_after_forward,
            )

        finally:
            if moe_hook_handle is not None:
                moe_hook_handle.remove()

    @torch.inference_mode()
    def _record_all_blocks_for_batch_group(
        self,
        data_batches: List[torch.Tensor],
        save_path: Optional[pathlib.Path] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process all blocks for a single batch group.

        Args:
            data_batches: List of input batches to process for this group
            save_path: Optional path to save intermediate results

        Returns:
            Dictionary mapping block numbers to their metrics
        """
        if not self.blocks:
            raise ValueError("No transformer blocks found in model")

        logger.info(
            f"Processing {len(self.blocks)} blocks with {len(data_batches)} batches"
        )

        self._capture_first_block_inputs(data_batches)

        for block_idx in range(len(self.blocks)):
            moe_module = self._find_moe_module_in_block(block_idx)
            if moe_module is None:
                logger.warning(f"No MoE module in block {block_idx}, forwarding only")
                self._forward_block(block_idx)
            else:
                self._record_activations_for_block(block_idx, moe_module=moe_module)

            # Save intermediate results
            if save_path:
                intermediate_path = save_path / f"block_{block_idx:03d}_metrics.pt"
                intermediate_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.state.get(block_idx, {}), intermediate_path)
                logger.info(f"Saved intermediate results to {intermediate_path}")

            cleanup_memory()

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.debug(
                    f"GPU memory after block {block_idx}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

        self.replay_cache.clear()
        cleanup_memory(synchronize=False)

        logger.info(f"Completed processing all {len(self.blocks)} blocks")
        return self.report_state()

    @torch.inference_mode()
    def record_all_blocks(
        self,
        data_batches: List[torch.Tensor],
        save_path: Optional[pathlib.Path] = None,
        batch_group_size: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process all blocks sequentially, optionally in groups of batches.

        Args:
            data_batches: List of input batches to process
            save_path: Optional path to save intermediate results
            batch_group_size: Optional maximum number of batches to cache and process
                per group. If None, all batches are processed in one pass.

        Returns:
            Dictionary mapping block numbers to their metrics
        """
        if batch_group_size is None or batch_group_size >= len(data_batches):
            return self._record_all_blocks_for_batch_group(data_batches, save_path)

        if batch_group_size < 1:
            raise ValueError("batch_group_size must be at least 1 when provided")

        total_groups = (len(data_batches) + batch_group_size - 1) // batch_group_size
        logger.info(
            "Processing %s blocks across %s batch groups of up to %s batches",
            len(self.blocks),
            total_groups,
            batch_group_size,
        )

        for group_idx, start in enumerate(range(0, len(data_batches), batch_group_size)):
            end = min(start + batch_group_size, len(data_batches))
            batch_group = data_batches[start:end]
            group_save_path = save_path
            if group_save_path is not None:
                group_save_path = group_save_path / f"group_{group_idx:03d}"

            logger.info(
                "Processing batch group %s/%s with %s batches",
                group_idx + 1,
                total_groups,
                len(batch_group),
            )
            self._record_all_blocks_for_batch_group(
                data_batches=batch_group,
                save_path=group_save_path,
            )
            cleanup_memory()

        return self.report_state()

    def report_state(self) -> Dict[int, Dict[str, Any]]:
        """
        Report the current state with OnlineStatsTracker converted to means.

        Returns:
            State dictionary with metrics per block
        """
        return {
            block_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in block_state.items()
            }
            for block_num, block_state in self.state.items()
        }

    def save_state(self, file_path: pathlib.Path):
        """Save the observer state to a file."""
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.report_state()

        # Move all tensors to CPU
        for block_num, block_state in state_dict.items():
            for key, value in block_state.items():
                if isinstance(value, torch.Tensor):
                    state_dict[block_num][key] = value.cpu()

        torch.save(state_dict, file_path)
        logger.info(f"State saved to {file_path}")

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        self._moe_modules_cache.clear()
        self.replay_cache.clear()
        cleanup_memory(synchronize=False)
        logger.debug("Observer state reset")

    def close_hooks(self):
        """Clean up resources."""
        self.reset()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.debug("Observer closed")
