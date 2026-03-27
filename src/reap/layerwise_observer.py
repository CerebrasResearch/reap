"""
Layerwise MoE Observer for memory-efficient expert pruning calibration.

This module implements a block-wise activation collection approach inspired by AutoGPTQ,
adapted for MoE expert pruning metrics (REAP, EAN, frequency, etc.).

Key features:
1. Only one transformer block is loaded on GPU at a time
2. Hidden states are cached between blocks (passed from block N to block N+1)
3. Streaming approach - batches are processed one at a time
4. Progressive loading/unloading of transformer blocks
5. Computes all REAP pruning metrics per layer

Memory optimization: Instead of loading the full model,
only loads embeddings + one block at a time.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import gc
import inspect
import logging
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from reap.metrics import OnlineStatsTracker
from reap.observer import (
    MoETransformerObserverConfig,
)
from reap.layerwise_model_utils import (
    extract_model_components,
    get_module_by_name,
    is_decoder_block,
    natural_sort_key,
    cleanup_memory,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerwiseMoEObserver:
    """
    Memory-efficient MoE observer that processes one transformer block at a time.

    This class collects the same pruning metrics as MoETransformerObserver but
    in a memory-efficient manner suitable for large models on single GPUs.

    Metrics collected per layer:
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
        Initialize the layerwise MoE observer.

        Args:
            model: The PyTorch MoE model to observe
            hook_config: Configuration for hooks (contains MoE-specific settings)
            block_names: List of transformer block names. Auto-detected if None.
        """
        self.model = model
        self.hook_config = hook_config

        # Auto-detect decoder blocks if not provided
        self.block_names = block_names or self._auto_detect_blocks()

        # Extract model components
        self.layers, self.outside_layer_modules = extract_model_components(
            self.model, self.block_names
        )

        # Cache for intermediate hidden states between blocks
        self.layer_inputs_cache: List[List[torch.Tensor]] = []
        self.layer_kwargs_cache: List[Dict] = []
        self.attention_masks_cache: List[torch.Tensor] = []
        self.position_ids_cache: List[torch.Tensor] = []

        # Track which block is currently loaded
        self.currently_loaded_block_idx = -1

        # Hooks for current block
        self.hooks = []

        # State dictionary to store metrics per layer
        self.state: Dict[int, Dict[str, Any]] = {}

        # MoE module cache per block
        self._moe_modules_cache: Dict[int, nn.Module] = {}

        # Forward signature cache per block
        self._forward_signature_cache: Dict[int, Tuple[set[str], bool]] = {}

        logger.info(
            f"LayerwiseMoEObserver initialized with {len(self.block_names)} blocks"
        )
        logger.info(
            f"Block names: {self.block_names[:3]}{'...' if len(self.block_names) > 3 else ''}"
        )

    def _auto_detect_blocks(self) -> List[str]:
        """Auto-detect decoder blocks in the model."""
        block_names = []

        for name, module in self.model.named_modules():
            if is_decoder_block(name, module):
                block_names.append(name)

        block_names.sort(key=natural_sort_key)

        if not block_names:
            logger.warning(
                "No decoder blocks detected. Falling back to modules with 'layer' in name."
            )
            for name, module in self.model.named_modules():
                if "layer" in name.lower() and isinstance(module, nn.Module):
                    block_names.append(name)

        return block_names

    def _find_moe_module_in_block(self, block_idx: int) -> Optional[nn.Module]:
        """Find the MoE module within a transformer block."""
        if block_idx in self._moe_modules_cache:
            return self._moe_modules_cache[block_idx]

        if not self.layers or block_idx >= len(self.layers):
            return None

        block = self.layers[block_idx]
        block_name = (
            self.block_names[block_idx]
            if block_idx < len(self.block_names)
            else f"layer_{block_idx}"
        )

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
        return None

    def _initialize_layer_state(self, num_experts: int) -> Dict[str, Any]:
        """Initialize state dictionary for a layer."""
        device = "cpu"
        layer_state = {}

        # Unnormalized counts
        layer_state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
        layer_state["expert_frequency"] = torch.zeros(
            num_experts, device=device, dtype=torch.long
        )
        layer_state["pairwise_expert_frequency"] = torch.zeros(
            num_experts, num_experts, dtype=torch.long, device=device
        )

        # Pruning metrics - EAN variants
        layer_state["ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["weighted_ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["ean_mean"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )
        layer_state["reap"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )

        # Weighted frequency
        layer_state["weighted_expert_frequency_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )

        # Super experts detection
        layer_state["max_activations"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float32, requires_grad=False
        )

        return layer_state

    def _safe_get_device(self, module: nn.Module) -> str:
        """Safely get the device of a module."""
        try:
            for param in module.parameters():
                device_str = str(param.device)
                if device_str != "meta":
                    return device_str
            for buffer in module.buffers():
                device_str = str(buffer.device)
                if device_str != "meta":
                    return device_str
            return "meta"
        except Exception:
            return "unknown"

    def _has_meta_tensors(self, module: nn.Module) -> bool:
        """Check if a module has any meta tensors."""
        try:
            for param in module.parameters():
                if str(param.device) == "meta":
                    return True
            for buffer in module.buffers():
                if str(buffer.device) == "meta":
                    return True
            return False
        except Exception:
            return False

    def _load_specific_block(self, block_idx: int) -> str:
        """Load only the specified transformer block, unloading others."""
        if self.currently_loaded_block_idx == block_idx:
            return self._safe_get_device(self.layers[block_idx])

        # Unload current block if any
        if self.currently_loaded_block_idx >= 0:
            self._unload_current_block()

        if self.layers and block_idx < len(self.layers):
            target_layer = self.layers[block_idx]

            try:
                has_meta = self._has_meta_tensors(target_layer)

                if has_meta:
                    logger.debug(
                        f"Block {block_idx} has meta tensors, skipping device move"
                    )
                else:
                    current_device = self._safe_get_device(target_layer)
                    if current_device == "cpu" and torch.cuda.is_available():
                        target_layer.to("cuda")
                        logger.debug(f"Moved block {block_idx} from CPU to CUDA")
            except Exception as e:
                logger.warning(f"Could not check/move block {block_idx}: {e}")

            final_device = self._safe_get_device(target_layer)
        else:
            final_device = "meta"

        self.currently_loaded_block_idx = block_idx
        logger.debug(f"Loaded block {block_idx}")

        return final_device

    def _unload_current_block(self):
        """Unload the currently loaded block to free memory."""
        if self.currently_loaded_block_idx < 0:
            return

        if self.layers and self.currently_loaded_block_idx < len(self.layers):
            current_layer = self.layers[self.currently_loaded_block_idx]

            try:
                has_meta = self._has_meta_tensors(current_layer)
                if has_meta:
                    logger.debug(
                        f"Block {self.currently_loaded_block_idx} has meta tensors, skipping CPU move"
                    )
                else:
                    current_layer.cpu()
                    logger.debug(
                        f"Unloaded block {self.currently_loaded_block_idx} to CPU"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not move block {self.currently_loaded_block_idx} to CPU: {e}"
                )

        self.currently_loaded_block_idx = -1
        cleanup_memory(synchronize=False)

    def _capture_first_layer_inputs(self, data_batches: List[torch.Tensor]):
        """
        Capture inputs to the first transformer layer.

        Args:
            data_batches: List of input token tensors
        """
        logger.info("Capturing inputs to first transformer layer...")

        # Clear previous captures
        self.layer_inputs_cache.clear()
        self.layer_kwargs_cache.clear()
        self.attention_masks_cache.clear()
        self.position_ids_cache.clear()

        if not self.layers or len(self.layers) == 0:
            raise ValueError("No transformer layers found")

        first_layer = self.layers[0]

        # Find primary device from embeddings
        primary_device = None
        model_dtype = None

        for module_name in self.outside_layer_modules:
            module = get_module_by_name(self.model, module_name)
            if module is not None:
                try:
                    for param in module.parameters():
                        if str(param.device) != "meta":
                            primary_device = param.device
                            model_dtype = param.dtype
                            break
                    if primary_device is not None:
                        break
                except Exception:
                    continue

        if primary_device is None:
            try:
                primary_device = next(first_layer.parameters()).device
                model_dtype = next(first_layer.parameters()).dtype
            except Exception:
                primary_device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

        logger.debug(
            f"Using primary device: {primary_device}, model dtype: {model_dtype}"
        )
        device = primary_device

        # Hook to capture layer inputs
        def store_input_hook(_, args, kwargs):
            layer_input = []
            for inp in args:
                layer_input.append(inp.detach().cpu())
            self.layer_inputs_cache.append(layer_input)

            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None:
                self.attention_masks_cache.append(attention_mask.detach().cpu())
            else:
                self.attention_masks_cache.append(None)

            position_ids = kwargs.get("position_ids", None)
            if position_ids is not None:
                self.position_ids_cache.append(position_ids.detach().cpu())
            else:
                self.position_ids_cache.append(None)

            layer_kwargs = {}
            cpu_device = torch.device("cpu")
            for k, v in kwargs.items():
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    layer_kwargs[k] = LayerwiseMoEObserver._move_to_device(
                        v, cpu_device
                    )
            layer_kwargs = LayerwiseMoEObserver._sanitize_cached_layer_kwargs(
                layer_kwargs
            )
            self.layer_kwargs_cache.append(layer_kwargs)

            raise ValueError("Input captured")

        handle = first_layer.register_forward_pre_hook(
            store_input_hook, with_kwargs=True
        )

        try:
            for batch in tqdm(data_batches, desc="Capturing layer inputs"):
                if isinstance(batch, torch.Tensor):
                    batch_dict = {"input_ids": batch.to(device)}
                elif isinstance(batch, dict) or isinstance(batch, BatchEncoding):
                    batch_dict = {}
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            if len(v.shape) == 1:
                                v = v.unsqueeze(0)
                            if torch.is_floating_point(v) and model_dtype is not None:
                                v = v.to(dtype=model_dtype)
                            batch_dict[k] = v.to(device)
                        else:
                            batch_dict[k] = v
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                try:
                    self.model(**batch_dict)
                except ValueError as e:
                    if "Input captured" in str(e):
                        continue
                    else:
                        raise
                except Exception:
                    raise
        finally:
            handle.remove()

        logger.info(f"Captured inputs for {len(self.layer_inputs_cache)} batches")

        if len(self.layer_inputs_cache) == 0:
            raise ValueError("Failed to capture any layer inputs")

    @staticmethod
    def _move_to_device(value, target_device: torch.device):
        """Recursively move tensors within nested structures to target device."""
        if torch.is_tensor(value) and str(value.device) != "meta":
            return value.to(target_device)
        elif isinstance(value, dict):
            return {
                k: LayerwiseMoEObserver._move_to_device(v, target_device)
                for k, v in value.items()
            }
        elif isinstance(value, (tuple, list)):
            moved = [
                LayerwiseMoEObserver._move_to_device(v, target_device) for v in value
            ]
            return type(value)(moved)
        return value

    @classmethod
    def _sanitize_cached_layer_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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

        signature = inspect.signature(self.layers[block_idx].forward)
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

    def _build_replay_kwargs(self, block_idx: int, layer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for replaying a decoder block without cache state."""
        replay_kwargs = self._sanitize_cached_layer_kwargs(layer_kwargs)
        accepted_kwargs, accepts_var_kwargs = self._get_forward_signature_info(block_idx)

        for key, value in self._REPLAY_KWARG_FORCED_VALUES.items():
            if accepts_var_kwargs or key in accepted_kwargs:
                replay_kwargs[key] = value

        if accepts_var_kwargs:
            return replay_kwargs

        return {key: value for key, value in replay_kwargs.items() if key in accepted_kwargs}

    def _prepare_block_inputs(
        self, batch_idx: int, target_device: torch.device
    ) -> Tuple[List[torch.Tensor], Dict]:
        """Prepare inputs for processing through a block."""
        layer_input = self.layer_inputs_cache[batch_idx]
        layer_kwargs = (
            self.layer_kwargs_cache[batch_idx]
            if batch_idx < len(self.layer_kwargs_cache)
            else {}
        )
        attention_mask = (
            self.attention_masks_cache[batch_idx]
            if batch_idx < len(self.attention_masks_cache)
            else None
        )
        position_ids = (
            self.position_ids_cache[batch_idx]
            if batch_idx < len(self.position_ids_cache)
            else None
        )

        # Move inputs to target device
        layer_input_gpu = []
        for inp in layer_input:
            if str(inp.device) != "meta":
                layer_input_gpu.append(inp.to(target_device))
            else:
                layer_input_gpu.append(inp)

        kwargs = dict(layer_kwargs)
        if attention_mask is not None and str(attention_mask.device) != "meta":
            kwargs["attention_mask"] = attention_mask.to(target_device)
        if position_ids is not None and str(position_ids.device) != "meta":
            kwargs["position_ids"] = position_ids.to(target_device)

        # Move all remaining kwargs to target device, including nested
        # structures like position_embeddings which is a (cos, sin) tuple.
        for k, v in kwargs.items():
            kwargs[k] = self._move_to_device(v, target_device)

        return layer_input_gpu, kwargs

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
        through a single MoE layer.

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

        # Initialize state for this layer if needed
        if block_idx not in self.state:
            self.state[block_idx] = self._initialize_layer_state(num_experts)

        # Compute activations for all experts
        activations = torch.zeros((num_experts, *flat_input.shape), device=device)

        if self.hook_config.fused_experts:
            # Fused experts (e.g., Llama-4)
            router_logits = moe_module.router(flat_input)
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
                router_logits = moe_module.gate(flat_input)
            elif hasattr(moe_module, "router"):
                router_logits = moe_module.router(flat_input)
            else:
                raise ValueError(
                    f"Cannot find router in MoE module at block {block_idx}"
                )

            _, selected_experts = torch.topk(router_logits, top_k, dim=-1)

            # Compute activations for all experts
            for idx, expert in enumerate(moe_module.experts):
                activations[idx] = expert(flat_input).to(device)

        # --- Compute pruning metrics ---
        # Filter by valid tokens (exclude padding) if attention mask is provided

        # Flatten selected_experts for processing: [batch*seq, top_k]
        selected_experts_flat = selected_experts.view(-1, top_k)

        if valid_token_mask is not None:
            # Count only valid (non-padding) tokens
            num_tokens = valid_token_mask.sum().item()
            num_tokens_tensor = torch.tensor(num_tokens, device="cpu", dtype=torch.long)

            # Filter to only valid tokens for frequency computation
            valid_selected_experts = selected_experts_flat[valid_token_mask]

            # Expert frequency - only count valid tokens
            expert_frequency = torch.bincount(
                valid_selected_experts.view(-1), minlength=num_experts
            ).to(device)
        else:
            # No mask - use all tokens
            num_tokens = batch_size * sequence_length
            num_tokens_tensor = torch.tensor(num_tokens, device="cpu", dtype=torch.long)

            # Expert frequency
            expert_frequency = torch.bincount(
                selected_experts_flat.view(-1), minlength=num_experts
            ).to(device)

        pairwise_expert_frequency = expert_frequency.unsqueeze(
            0
        ) + expert_frequency.unsqueeze(1)

        # Update counts
        self.state[block_idx]["total_tokens"] += num_tokens_tensor
        self.state[block_idx]["expert_frequency"] += expert_frequency.to(
            "cpu", torch.long
        )
        self.state[block_idx]["pairwise_expert_frequency"] += (
            pairwise_expert_frequency.to("cpu", torch.long)
        )

        # Pruning criteria
        ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
        ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
        weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
        reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
        weighted_expert_frequency_sum = torch.zeros(
            num_experts, device=device, dtype=torch.float64
        )

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(device)
        prior_max_activations = self.state[block_idx]["max_activations"]

        # Renormalize router weights if configured
        if self.hook_config.renormalize_router_weights:
            topk_weights = torch.gather(routing_weights, 1, selected_experts_flat)
            routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
            routing_weights = torch.clamp(
                routing_weights, min=torch.finfo(routing_weights.dtype).eps
            )

        for i in range(num_experts):
            # Compute which tokens selected this expert (among their top_k choices)
            active_mask = (selected_experts_flat == i).any(dim=-1).to(device)

            # If we have a valid token mask, also require the token to be valid
            if valid_token_mask is not None:
                active_mask = active_mask & valid_token_mask

            if not active_mask.any():
                continue

            active_router_weights = routing_weights[active_mask, i]
            ean_norm = torch.linalg.norm(activations[i, active_mask, :], dim=-1)

            ean_sum[i] = ean_norm.sum().to(device)
            ean_mean[i] = ean_norm.mean().to(device)
            weighted_expert_frequency_sum[i] = active_router_weights.sum().to(device)
            weighted_ean_sum[i] = (ean_norm * active_router_weights).sum().to(device)
            reap[i] = (ean_norm * active_router_weights).mean().to(device)

            # Super experts detection
            selected_activations = activations[i, active_mask, :]
            selected_activations_max = selected_activations.max().to(device="cpu")
            if selected_activations_max > prior_max_activations[i]:
                self.state[block_idx]["max_activations"][i] = selected_activations_max
                prior_max_activations[i] = selected_activations_max

        # Update state
        self.state[block_idx]["ean_sum"] += ean_sum.to(device="cpu")
        self.state[block_idx]["ean_mean"].update(
            ean_mean.to("cpu"), expert_frequency.to("cpu")
        )
        self.state[block_idx]["weighted_ean_sum"] += weighted_ean_sum.to(device="cpu")
        self.state[block_idx]["reap"].update(reap.to("cpu"), expert_frequency.to("cpu"))
        self.state[block_idx]["weighted_expert_frequency_sum"] += (
            weighted_expert_frequency_sum.to(device="cpu")
        )

        # Clean up
        del activations, selected_experts, selected_experts_flat, router_logits
        del expert_frequency, pairwise_expert_frequency, routing_weights
        if valid_token_mask is not None:
            del valid_token_mask
        gc.collect()

    @torch.inference_mode()
    def collect_activations_for_block(
        self,
        block_idx: int,
        data_batches: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Collect MoE activations and compute metrics for a single block.

        Args:
            block_idx: Index of the block to process
            data_batches: List of input batches

        Returns:
            Dictionary with computed metrics for this block
        """
        if block_idx == 0:
            self._capture_first_layer_inputs(data_batches)

        block_name = (
            self.block_names[block_idx]
            if block_idx < len(self.block_names)
            else f"layer_{block_idx}"
        )
        logger.info(
            f"Processing block {block_idx + 1}/{len(self.layers)}: {block_name}"
        )

        self.model.eval()

        if not self.layers or block_idx >= len(self.layers):
            raise ValueError(f"Block {block_idx} not found")

        device_str = self._load_specific_block(block_idx)
        layer = self.layers[block_idx]

        if device_str == "meta":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device_str)

        # Find MoE module in this block
        moe_module = self._find_moe_module_in_block(block_idx)
        captured_moe_input: Dict[str, torch.Tensor] = {}

        if moe_module is None:
            logger.warning(
                f"No MoE module found in block {block_idx}; continuing without MoE metrics"
            )
        else:
            # Register a forward hook on the MoE module to capture its input
            def _capture_moe_input_hook(module, args, output):
                captured_moe_input["input"] = args[0].detach()
                return output

            moe_hook_handle = moe_module.register_forward_hook(_capture_moe_input_hook)

        try:
            if not self.layer_inputs_cache:
                raise ValueError("No cached layer inputs available")

            num_batches = min(len(self.layer_inputs_cache), len(data_batches))
            layer_outputs = []

            for batch_idx in tqdm(range(num_batches), desc=f"Processing {block_name}"):
                layer_input, layer_kwargs = self._prepare_block_inputs(
                    batch_idx, target_device
                )
                attention_mask = layer_kwargs.get("attention_mask", None)

                layer_kwargs = self._build_replay_kwargs(block_idx, layer_kwargs)
                captured_moe_input.clear()

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    outputs = layer(*layer_input, **layer_kwargs)

                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs

                if moe_module is not None:
                    moe_input = captured_moe_input["input"]
                    
                    self._process_moe_activations(
                        block_idx,
                        moe_module,
                        moe_input,
                        target_device,
                        attention_mask=attention_mask,
                    )

                layer_outputs.append([hidden_states.detach().cpu()])

                del outputs, hidden_states, layer_input, layer_kwargs
                if moe_module is not None:
                    del moe_input
                captured_moe_input.clear()

                if batch_idx % 4 == 0:
                    cleanup_memory(synchronize=False)

            # Update cache for next layer
            if block_idx < len(self.layers) - 1:
                del self.layer_inputs_cache
                self.layer_inputs_cache = layer_outputs

            logger.info(f"Completed block {block_idx}: processed {num_batches} batches")

            return self.state.get(block_idx, {})

        finally:
            if moe_hook_handle is not None:
                moe_hook_handle.remove()
            self._unload_current_block()

    @torch.inference_mode()
    def _collect_all_blocks_for_batch_group(
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
            Dictionary mapping layer numbers to their metrics
        """
        if not self.layers:
            raise ValueError("No transformer layers found in model")

        logger.info(
            f"Processing {len(self.layers)} blocks with {len(data_batches)} batches"
        )

        for block_idx in range(len(self.layers)):
            self.collect_activations_for_block(block_idx, data_batches)

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

        # Clear caches
        self._clear_cache()

        logger.info(f"Completed processing all {len(self.layers)} blocks")
        return self.report_state()

    @torch.inference_mode()
    def collect_all_blocks(
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
            Dictionary mapping layer numbers to their metrics
        """
        if batch_group_size is None or batch_group_size >= len(data_batches):
            return self._collect_all_blocks_for_batch_group(data_batches, save_path)

        if batch_group_size < 1:
            raise ValueError("batch_group_size must be at least 1 when provided")

        total_groups = (len(data_batches) + batch_group_size - 1) // batch_group_size
        logger.info(
            "Processing %s blocks across %s batch groups of up to %s batches",
            len(self.layers),
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
            self._collect_all_blocks_for_batch_group(
                data_batches=batch_group,
                save_path=group_save_path,
            )
            cleanup_memory()

        return self.report_state()

    def _clear_cache(self):
        """Clear all cached data to free memory."""
        self.layer_inputs_cache.clear()
        self.layer_kwargs_cache.clear()
        self.attention_masks_cache.clear()
        self.position_ids_cache.clear()
        cleanup_memory(synchronize=False)

    def report_state(self) -> Dict[int, Dict[str, Any]]:
        """
        Report the current state with OnlineStatsTracker converted to means.

        Returns:
            State dictionary with metrics per layer
        """
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

    def save_state(self, file_path: pathlib.Path):
        """Save the observer state to a file."""
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.report_state()

        # Move all tensors to CPU
        for layer_num, layer_state in state_dict.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    state_dict[layer_num][key] = value.cpu()

        torch.save(state_dict, file_path)
        logger.info(f"State saved to {file_path}")

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        self._moe_modules_cache.clear()
        self._clear_cache()
        logger.debug("Observer state reset")

    def close_hooks(self):
        """Clean up resources."""
        self.reset()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.debug("Observer closed")
