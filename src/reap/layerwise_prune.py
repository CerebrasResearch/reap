"""
Layerwise Expert Pruning for MoE Models.

This module provides a memory-efficient entry point for expert pruning
that processes the model one layer at a time, enabling calibration of
large MoE models on a single GPU.

Key differences from standard prune.py:
1. Model is loaded on CPU with device_map="cpu"
2. Only one transformer block is on GPU at a time
3. Hidden states are cached between blocks
4. Significantly reduced GPU memory requirements

Usage:
    python -m reap.layerwise_prune \
        --model_name "Qwen/Qwen3-30B-A3B" \
        --dataset_name "theblackcat102/evol-codealpaca-v1" \
        --prune_method "reap" \
        --compression_ratio 0.5 \
        --batch_size 4
"""

from __future__ import annotations
import logging
import dataclasses
import pathlib
import hashlib
from typing import Any, Dict, List
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed

from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
    LayerwiseArgs,
)
from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.model_util import patched_model_map
from reap.observer import OBSERVER_CONFIG_REGISTRY
from reap.layerwise_observer import LayerwiseMoEObserver
from reap.layerwise_model_utils import cleanup_memory
from reap.eval import run_evaluate
from reap.prune import prune as prune_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    """Create a clean directory name from model and dataset names."""
    import re

    def str_to_directory_name(s: str) -> str:
        return re.sub(r"[^\w\-_.]", "_", s)

    model_clean = model_name.split("/")[-1]
    model_clean = str_to_directory_name(model_clean)

    if "," in dataset_name:
        spec_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        dataset_clean = f"composite_{spec_hash}"
        logger.info(
            f"Composite dataset spec detected. Using directory name: {dataset_clean}"
        )
    else:
        dataset_clean = dataset_name.split("/")[-1]
        dataset_clean = str_to_directory_name(dataset_clean)

    results_dir = pathlib.Path("./artifacts") / model_clean / dataset_clean

    if results_dir.exists():
        logger.warning(f"Directory '{results_dir}' already exists")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created artifacts directory: {results_dir}")

    return results_dir


def _get_observer_output_path(
    results_dir: pathlib.Path,
    dataset_name: str,
    output_file_name: str,
) -> pathlib.Path:
    if (
        dataset_name == "combined"
        or parse_composite_dataset_spec(dataset_name) is not None
    ):
        return results_dir / "all" / output_file_name
    return results_dir / "layerwise" / output_file_name


def prepare_calibration_batches(
    tokenizer,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
) -> List[torch.Tensor]:
    """
    Prepare calibration samples for layerwise processing.

    Returns a list of tokenized input tensors.
    """
    logger.info(f"Loading dataset {ds_args.dataset_name}...")

    composite_components = parse_composite_dataset_spec(
        ds_args.dataset_name, default_split=ds_args.split
    )

    if composite_components is not None:
        all_batches = []
        total_samples = sum(component.num_batches for component in composite_components)
        logger.info(
            f"Composite dataset specified, overwriting given batches_per_category={obs_args.batches_per_category} "
            f"with values in composite dataset spec."
        )
        logger.info(
            f"Preparing composite calibration data with {len(composite_components)} "
            f"components, {total_samples} total samples."
        )

        for component in composite_components:
            comp_label = f"{component.name}[{component.split}]"
            logger.info(
                f"Loading composite component {comp_label} ({component.num_batches} batches)"
            )
            category_data_batches = load_category_batches(
                dataset_name=component.name,
                split=component.split,
                subset=component.subset,
                tokenizer=tokenizer,
                model_max_length=obs_args.model_max_length,
                split_by_category=False,
                return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
                truncate=obs_args.truncate,
                batches_per_category=component.num_batches,
                batch_size=obs_args.batch_size,
            )
            for category, batches in category_data_batches.items():
                all_batches.extend(batches)
                logger.info(f"Added {len(batches)} batches from category: {category}")

        logger.info(f"Total calibration batches: {len(all_batches)}")
        return all_batches

    category_data_batches = load_category_batches(
        dataset_name=ds_args.dataset_name,
        split=ds_args.split,
        subset=ds_args.dataset_config_name,
        tokenizer=tokenizer,
        model_max_length=obs_args.model_max_length,
        split_by_category=obs_args.split_by_category,
        return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
        truncate=obs_args.truncate,
        batches_per_category=obs_args.batches_per_category,
        batch_size=obs_args.batch_size,
    )

    # Flatten all batches into a single list
    all_batches = []
    for category, samples in category_data_batches.items():
        all_batches.extend(samples)
        logger.info(f"Added {len(samples)} samples from category: {category}")

    logger.info(f"Total calibration samples: {len(all_batches)}")
    return all_batches


def record_activations_layerwise(
    model,
    tokenizer,
    data_batches: List[torch.Tensor],
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    layerwise_args: LayerwiseArgs,
    results_dir: pathlib.Path,
) -> Dict[int, Dict[str, Any]]:
    """
    Record MoE activations using layerwise processing.

    This function processes the model one block at a time to minimize
    GPU memory usage.
    """
    logger.info("Starting layerwise activation recording...")

    # Create observer config from model-specific settings
    model_class_name = model.__class__.__name__
    if model_class_name not in OBSERVER_CONFIG_REGISTRY:
        raise ValueError(
            f"No observer configuration for model '{model_class_name}'. "
            f"Supported: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )

    hook_config = OBSERVER_CONFIG_REGISTRY[model_class_name](
        renormalize_router_weights=obs_args.renormalize_router_weights,
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
    )

    # Create layerwise observer
    observer = LayerwiseMoEObserver(
        model=model,
        hook_config=hook_config,
    )

    # Process all blocks
    save_path = (
        _get_observer_output_path(
            results_dir,
            ds_args.dataset_name,
            obs_args.output_file_name,
        ).parent
        / "layerwise_intermediate"
        if layerwise_args.save_intermediate
        else None
    )

    observer_data = observer.record_all_blocks(
        data_batches=data_batches,
        save_path=save_path,
        batch_group_size=layerwise_args.batch_group_size,
    )

    # Save complete state
    output_file = _get_observer_output_path(
        results_dir,
        ds_args.dataset_name,
        obs_args.output_file_name,
    )
    observer.save_state(output_file)

    logger.info(f"Layerwise activation recording complete. Saved to {output_file}")

    return observer_data


def get_pruned_model_dir(
    results_dir: pathlib.Path,
    n_experts_to_prune: int,
    total_experts: int,
    prune_args: PruneArgs,
    seed: int,
    renorm: bool,
) -> pathlib.Path:
    """Generate output directory path for pruned model."""
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
    pruned_model_name = f"layerwise_{prune_args.prune_method}"

    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"

    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")

    return pruned_model_dir


def dump_args_to_yaml(
    pruned_model_dir: pathlib.Path,
    **all_args,
):
    """Dump all arguments to a YAML file."""

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = {}
    for name, arg in all_args.items():
        if dataclasses.is_dataclass(arg):
            serializable_args[name] = convert_paths_to_str(dataclasses.asdict(arg))
        else:
            serializable_args[name] = convert_paths_to_str(arg)

    output_path = pruned_model_dir / "reap_layerwise_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"Arguments saved to {output_path}")


def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
            LayerwiseArgs,
        )
    )
    (
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        prune_args,
        cluster_args,
        layerwise_args,
    ) = parser.parse_args_into_dataclasses()

    # Validation
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError(
            "Only one of perserve_super_experts or perserve_outliers can be True."
        )
    if (
        layerwise_args.batch_group_size is not None
        and layerwise_args.batch_group_size < 1
    ):
        raise ValueError("layerwise batch_group_size must be at least 1 when provided.")

    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # Get patched model name if needed
    model_name = patched_model_map(model_args.model_name)

    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = None

    cached_data_path = _get_observer_output_path(
        results_dir,
        ds_args.dataset_name,
        obs_args.output_file_name,
    )

    if ds_args.dataset_name == "combined":
        if cached_data_path.exists():
            logger.info(f"Loading cached observer data from {cached_data_path}")
            observer_data = torch.load(cached_data_path, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {cached_data_path}"
            )
    else:
        # Prepare calibration samples
        logger.info("Preparing calibration samples...")
        data_batches = prepare_calibration_batches(tokenizer, ds_args, obs_args)

        # Load model on CPU for layerwise processing
        logger.info(f"Loading model {model_name} on CPU for layerwise processing...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=layerwise_args.low_cpu_mem_usage,
        )
        model.eval()

        logger.info(f"Model loaded: {model.__class__.__name__}")
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {num_params / 1e9:.2f}B")

        # Check for cached observer data
        if cached_data_path.exists() and not obs_args.overwrite_observations:
            logger.info(f"Loading cached observer data from {cached_data_path}")
            observer_data = torch.load(cached_data_path, weights_only=False)
        else:
            # Record activations using layerwise processing
            logger.info("Recording activations using layerwise processing...")
            observer_data = record_activations_layerwise(
                model,
                tokenizer,
                data_batches,
                ds_args,
                obs_args,
                layerwise_args,
                results_dir,
            )

    if reap_args.run_observer_only:
        logger.info("Observer run completed. Exiting (run_observer_only=True)")
        return

    # Calculate number of experts to prune
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set."
            )
        total_experts = len(
            observer_data[next(iter(observer_data))]["expert_frequency"]
        )
        n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
        logger.info(
            f"Calculated n_experts_to_prune: {n_experts_to_prune} "
            f"(compression_ratio: {cluster_args.compression_ratio})"
        )
    else:
        total_experts = len(
            observer_data[next(iter(observer_data))]["expert_frequency"]
        )

    # Get output directory
    pruned_model_dir = get_pruned_model_dir(
        results_dir,
        n_experts_to_prune,
        total_experts,
        prune_args,
        reap_args.seed,
        obs_args.renormalize_router_weights,
    )

    # Check if already pruned
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model already exists at {pruned_model_dir}. Skipping pruning."
        )
    else:
        # Reload model on auto device for pruning
        logger.info("Reloading model on GPU for pruning...")
        if model is not None:
            del model
        cleanup_memory()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        # Prune
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune_model(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )

        # Save tokenizer
        tokenizer.save_pretrained(pruned_model_dir)

        # Save args
        dump_args_to_yaml(
            pruned_model_dir,
            reap_args=reap_args,
            ds_args=ds_args,
            obs_args=obs_args,
            model_args=model_args,
            eval_args=eval_args,
            prune_args=prune_args,
            cluster_args=cluster_args,
            layerwise_args=layerwise_args,
        )

        logger.info("Pruning completed successfully!")

    # Evaluation
    if reap_args.do_eval:
        logger.info("Starting evaluation...")
        if model is not None:
            del model
        del observer_data
        cleanup_memory()

        model_args.model_name = pruned_model_dir
        run_evaluate(
            model_args,
            pruned_model_dir / "eval",
            eval_args,
            reap_args.seed,
        )


if __name__ == "__main__":
    main()
