"""
Model utilities for layerwise processing of MoE models.

"""

from __future__ import annotations
import re
from typing import List, Tuple, Union, Optional
import gc
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def natural_sort_key(value: str) -> list[object]:
    """Sort strings containing numbers in human order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def cleanup_memory(synchronize: bool = True) -> None:
    """
    Clean up memory by running garbage collection and clearing CUDA cache.

    Args:
        synchronize: If True, also synchronize CUDA operations before cleanup
    """
    gc.collect()
    if torch.cuda.is_available():
        if synchronize:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def is_linear_like(module: nn.Module) -> bool:
    """
    Heuristic to detect projection/expert modules that behave like linear layers.

    Args:
        module: PyTorch module to check

    Returns:
        True if the module behaves like a linear layer
    """
    # Exclude Embeddings
    if isinstance(module, nn.Embedding):
        return False

    # Standard Linear
    if isinstance(module, nn.Linear):
        return True

    # 1x1 Conv used as linear projection
    if isinstance(module, nn.Conv1d):
        try:
            if tuple(module.kernel_size) == (1,):
                return True
        except Exception:
            pass

    # Modules with 2D weight Parameters (common in custom expert modules)
    try:
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and hasattr(param, "dim") and param.dim() == 2:
                if param_name in {
                    "weight",
                    "w1",
                    "w2",
                    "w3",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                    "gate_up_proj",
                    "expert_weight",
                }:
                    return True
                class_name = module.__class__.__name__.lower()
                if any(
                    k in class_name for k in ("expert", "mlp", "ffn", "feedforward")
                ):
                    return True
    except Exception:
        pass

    return False


def is_decoder_block(name: str, module: nn.Module) -> bool:
    """
    Check if a module is a decoder block.

    Args:
        name: Full name of the module in the model
        module: The module to check

    Returns:
        True if this is a decoder/transformer block
    """
    # Common patterns for individual transformer decoder blocks
    numbered_patterns = [
        r"\.layers\.\d+$",  # model.layers.0, model.layers.1, etc.
        r"\.decoder\.layers\.\d+$",  # model.decoder.layers.0, etc.
        r"\.transformer\.h\.\d+$",  # model.transformer.h.0, etc. (GPT-style)
        r"\.transformer\.layers\.\d+$",  # model.transformer.layers.0, etc.
        r"\.decoder\.block\.\d+$",  # model.decoder.block.0, etc. (T5-style)
    ]

    for pattern in numbered_patterns:
        if re.search(pattern, name):
            # Also check that it contains projection-like layers
            if any(is_linear_like(child) for child in module.modules()):
                return True

    return False


def get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    """
    Get a module by its full dotted name.

    Args:
        model: The root model
        module_name: Dotted path to the module (e.g., "model.layers.0.mlp")

    Returns:
        The module if found, None otherwise
    """
    parts = module_name.split(".")
    module = model
    try:
        for part in parts:
            module = getattr(module, part)
        return module
    except AttributeError:
        return None


def extract_model_components(
    model: nn.Module, block_names: List[str]
) -> Tuple[Union[nn.Module, List[nn.Module]], List[str]]:
    """
    Extract and cache essential model components for selective loading.

    This function identifies the transformer layers container and modules
    that exist outside the transformer blocks (embeddings, layer norms, etc.).

    Args:
        model: The model to extract components from
        block_names: List of decoder block names

    Returns:
        Tuple of (layers, outside_layer_modules) where:
        - layers: ModuleList/Sequential container or list of individual modules
        - outside_layer_modules: List of module names outside transformer blocks
    """
    logger.info("Extracting model components for optimized loading...")

    # Find the transformer layers container
    layers_found = False
    layers = None

    for name, module in model.named_modules():
        if any(block_name.startswith(name + ".") for block_name in block_names):
            # This is the container holding all transformer layers
            if hasattr(module, "__len__") and len(module) > 0:
                layers = module
                layers_found = True
                logger.info(
                    f"Found transformer layers container: {name} with {len(module)} layers"
                )
                break

    if not layers_found:
        # Fallback: collect individual layers
        layers_list = []
        for block_name in block_names:
            for name, module in model.named_modules():
                if name == block_name:
                    layers_list.append(module)
                    break
        layers = layers_list
        logger.info(f"Collected {len(layers_list)} individual transformer layers")

    # Detect outside layer modules (embeddings, layer norm, etc.)
    outside_layer_modules = []

    # Common patterns for modules outside transformer blocks
    outside_patterns = [
        "embed_tokens",
        "wte",
        "word_embeddings",
        "embeddings.word_embeddings",
        "embed_positions",
        "wpe",
        "position_embeddings",
        "norm",
        "ln_f",
        "final_layer_norm",
        "layer_norm",
        "lm_head",
        "embed_out",
        "output_projection",
    ]

    for name, module in model.named_modules():
        # Skip if this is part of a transformer block
        if any(name.startswith(block_name) for block_name in block_names):
            continue

        # Check if this looks like an outside module
        if any(pattern in name for pattern in outside_patterns):
            outside_layer_modules.append(name)
            logger.debug(f"Found outside module: {name}")

    return layers, outside_layer_modules
