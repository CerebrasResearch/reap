# Local addition (not upstream REAP): streams a checkpoint's real tensors from its
# safetensors shards directly onto a target device, on demand, one module at a time.
#
# Used to make LayerwiseMoEObserver's block replay genuinely disk-streaming: the
# model is constructed with meta-device weights (accelerate.init_empty_weights(),
# ~0 RAM), and a block's real tensors are materialized here right before its
# forward and released back to meta right after -- instead of REAP's original
# approach of loading the entire model onto CPU via device_map="cpu" (~794GB of RAM
# for a model like Ornith-1.0-397B, not available on hardware with far less RAM+VRAM
# combined than the checkpoint size).
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

logger = logging.getLogger(__name__)


class SafetensorsIndex:
    """Lazy, mmap-backed access to a checkpoint's tensors by name.

    Deliberately does NOT cache open safe_open() handles across calls: mmap'd pages
    stay resident (counted in RSS) for as long as the mapping is open, even after
    the torch tensor copied out of them is freed. For a checkpoint far larger than
    available RAM, caching handles indefinitely would silently re-create the exact
    problem this module exists to avoid (RSS creeping up by however much of the
    file has been touched so far, instead of staying bounded to one block at a
    time). Every read (or batch of reads via read_tensors) opens a shard, reads,
    and closes/unmaps before returning.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            # Small, unsharded checkpoint: one model.safetensors file.
            single_file = self.checkpoint_dir / "model.safetensors"
            with safe_open(str(single_file), framework="pt") as f:
                self.weight_map = {name: single_file.name for name in f.keys()}

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map

    def read_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        return self.read_tensors([name], device=device)[name]

    def read_tensors(self, names: list[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Read several tensors, grouped by shard file so each shard is opened and
        closed (unmapped) once regardless of how many tensors are pulled from it."""
        by_shard: Dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)

        result: Dict[str, torch.Tensor] = {}
        for shard_name, shard_tensor_names in by_shard.items():
            with safe_open(str(self.checkpoint_dir / shard_name), framework="pt") as f:
                for name in shard_tensor_names:
                    tensor = f.get_tensor(name)
                    if device != "cpu":
                        tensor = tensor.to(device)
                    result[name] = tensor
        return result

    def tensor_names_with_prefix(self, prefix: str) -> list[str]:
        dotted = prefix if prefix.endswith(".") else prefix + "."
        return [n for n in self.weight_map if n == prefix or n.startswith(dotted)]


def materialize_module(
    module: nn.Module, module_name: str, index: SafetensorsIndex, device: str
) -> None:
    """Populate `module`'s (currently meta) parameters/buffers with real data read
    directly from the checkpoint, onto `device`. `module_name` is `module`'s dotted
    path in the full model (used as the tensor-name prefix in the checkpoint)."""
    targets = []  # (param_name, full_checkpoint_name)
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device) != "meta":
            continue  # already materialized (e.g. shared/tied weights)
        full_name = f"{module_name}.{name}"
        if not index.has_tensor(full_name):
            logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, full_name))

    if not targets:
        return
    values = index.read_tensors([full_name for _, full_name in targets], device=device)
    for name, full_name in targets:
        set_module_tensor_to_device(module, name, device, value=values[full_name])


def free_module(module: nn.Module) -> None:
    """Release a module's real tensors back to the meta device, freeing memory."""
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device) == "meta":
            continue
        set_module_tensor_to_device(module, name, "meta")
