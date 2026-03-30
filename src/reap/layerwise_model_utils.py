"""
Model utilities for layerwise processing of MoE models.

"""

from __future__ import annotations
import re
from typing import Any, List, Tuple, Union, Optional
import gc
import logging

import torch
import torch.nn as nn
from itertools import chain

logger = logging.getLogger(__name__)


NATURAL_SORT_RE = re.compile(r"(\d+)")

LINEAR_PARAM_NAMES = frozenset(
    {
        "weight",
        "w1",
        "w2",
        "w3",
        "up_proj",
        "down_proj",
        "gate_proj",
        "gate_up_proj",
        "expert_weight",
    }
)

LINEAR_CLASS_HINTS = ("expert", "mlp", "ffn", "feedforward")

DECODER_BLOCK_PATTERNS = (
    re.compile(r"\.layers\.\d+$"),
    re.compile(r"\.decoder\.layers\.\d+$"),
    re.compile(r"\.transformer\.h\.\d+$"),
    re.compile(r"\.transformer\.layers\.\d+$"),
    re.compile(r"\.decoder\.block\.\d+$"),
)

NON_BACKBONE_PATTERNS = (
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
)


def _iter_module_tensors(module: nn.Module):
    """Yield all parameters and buffers from a module."""
    yield from module.parameters()
    yield from module.buffers()


def safe_get_device(module: nn.Module) -> str:
    """Safely get the first non-meta device of a module."""
    try:
        for tensor in _iter_module_tensors(module):
            device_str = str(tensor.device)
            if device_str != "meta":
                return device_str
        return "meta"
    except Exception:
        raise


def has_meta_tensors(module: nn.Module) -> bool:
    """Check whether a module contains any meta tensors."""
    try:
        return any(str(tensor.device) == "meta" for tensor in _iter_module_tensors(module))
    except Exception:
        return False


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Return a key that sorts strings with embedded numbers naturally.

    Example:
        "file2" < "file10"
    """
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in NATURAL_SORT_RE.split(value)
    )


def cleanup_memory(synchronize: bool = True) -> None:
    """Run Python GC and release cached CUDA memory when available."""
    gc.collect()

    if not torch.cuda.is_available():
        return

    if synchronize:
        torch.cuda.synchronize()

    torch.cuda.empty_cache()


def move_to_device(value: Any, target_device: torch.device) -> Any:
    """Recursively move tensors within nested structures to target device."""
    if torch.is_tensor(value) and str(value.device) != "meta":
        return value.to(target_device)
    if isinstance(value, dict):
        return {k: move_to_device(v, target_device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        moved = [move_to_device(v, target_device) for v in value]
        return type(value)(moved)
    return value


def is_linear_like(module: nn.Module) -> bool:
    """
    Heuristically detect modules that behave like linear projections.
    """
    if isinstance(module, nn.Embedding):
        return False

    if isinstance(module, nn.Linear):
        return True

    if _is_pointwise_conv1d(module):
        return True

    return _has_linear_like_local_weights(module)


def _is_pointwise_conv1d(module: nn.Module) -> bool:
    """Return True for 1x1 Conv1d layers used as projections."""
    return isinstance(module, nn.Conv1d) and module.kernel_size == (1,)


def _has_linear_like_local_weights(module: nn.Module) -> bool:
    """
    Heuristic for custom modules that expose 2D local parameters
    resembling projection weights.
    """
    local_2d_param_names = {
        name
        for name, param in module.named_parameters(recurse=False)
        if param is not None and getattr(param, "ndim", None) == 2
    }

    if not local_2d_param_names:
        return False

    if local_2d_param_names & LINEAR_PARAM_NAMES:
        return True

    class_name = module.__class__.__name__.casefold()
    return any(hint in class_name for hint in LINEAR_CLASS_HINTS)


def _matches_decoder_block_name(name: str) -> bool:
    return any(pattern.search(name) for pattern in DECODER_BLOCK_PATTERNS)


def _has_projection_like_child(module: nn.Module) -> bool:
    return any(is_linear_like(child) for child in module.modules())


def is_decoder_block(name: str, module: nn.Module) -> bool:
    """
    Return True if the module looks like a transformer decoder block.
    """
    return _matches_decoder_block_name(name) and _has_projection_like_child(module)


def get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    module = model

    for part in module_name.split("."):
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and hasattr(module, "__getitem__"):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            return None

    return module


def _is_same_or_child(name: str, parent: str) -> bool:
    """Return True if `name` is `parent` or a descendant of it."""
    return name == parent or name.startswith(f"{parent}.")


def _is_non_empty_container(module: nn.Module) -> bool:
    """Best-effort check for ModuleList/Sequential-like containers."""
    try:
        return len(module) > 0  # type: ignore[arg-type]
    except TypeError:
        return False


def _find_blocks_container(
    modules_by_name: Dict[str, nn.Module],
    block_names: Sequence[str],
) -> Union[nn.Module, None]:
    """
    If all blocks share the same direct parent, use that as the container.
    Example:
        transformer.h.0, transformer.h.1 -> transformer.h
    """
    parent_names = {
        block_name.rsplit(".", 1)[0]
        for block_name in block_names
        if "." in block_name
    }

    if len(parent_names) != 1:
        return None

    container_name = next(iter(parent_names))
    container = modules_by_name.get(container_name)

    if container is not None and _is_non_empty_container(container):
        logger.info(
            "Found transformer blocks container: %s with %d blocks",
            container_name,
            len(container),
        )
        return container

    return None


def _find_non_backbone_modules(
    module_names: Sequence[str],
    block_names: Sequence[str],
) -> List[str]:
    block_name_set = set(block_names)
    block_prefixes = tuple(f"{name}." for name in block_names)

    non_backbone_modules: List[str] = []

    for name in module_names:
        # Skip transformer blocks and anything nested inside them.
        if name in block_name_set or name.startswith(block_prefixes):
            continue

        # Prefer exact/suffix matching over raw substring matching.
        if any(name == pattern or name.endswith(f".{pattern}") for pattern in NON_BACKBONE_PATTERNS):
            non_backbone_modules.append(name)
            logger.debug("Found non-backbone module: %s", name)

    return non_backbone_modules


def extract_model_components(
    model: nn.Module,
    block_names: List[str],
):
    """
    Extract and cache essential model components for selective loading.

    Returns:
        (blocks, non_backbone_modules)
    """
    if not block_names:
        raise ValueError("block_names must not be empty")

    logger.info("Extracting model components...")

    modules_by_name = dict(model.named_modules())

    blocks = _find_blocks_container(modules_by_name, block_names)
    if blocks is None:
        blocks = [modules_by_name[name] for name in block_names if name in modules_by_name]
        logger.info("Collected %d individual transformer blocks", len(blocks))

    non_backbone_modules = _find_non_backbone_modules(
        list(modules_by_name.keys()),
        block_names,
    )

    return blocks, non_backbone_modules


def find_decoder_blocks(model: nn.Module) -> List[str]:
    """Detect decoder blocks in the model."""
    modules = list(model.named_modules())

    block_names = [name for name, module in modules if is_decoder_block(name, module)]
    if block_names:
        return sorted(block_names, key=natural_sort_key)

    raise RuntimeError(
        "No decoder blocks detected in the model."
    )
