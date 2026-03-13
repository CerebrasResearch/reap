from __future__ import annotations
import time
import logging
import dataclasses
import pathlib
import time
from typing import Any
import gc
import yaml

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.main import record_activations, smoke_test, create_results_directory
from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
)
from reap.data import DATASET_REGISTRY
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
)
from reap.model_util import get_moe, assert_merge, MODEL_ATTRS, patched_model_map, get_super_expert_indices
from reap.eval import run_evaluate
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def dump_args_to_yaml(
    pruned_model_dir: pathlib.Path,
    reap_args: ReapArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    model_args: ModelArgs,
    eval_args: EvalArgs,
    prune_args: PruneArgs,
    cluster_args: ClusterArgs,
):
    """Dump all arguments to a YAML file."""
    all_args = {
        "reap_args": dataclasses.asdict(reap_args),
        "ds_args": dataclasses.asdict(ds_args),
        "obs_args": dataclasses.asdict(obs_args),
        "model_args": dataclasses.asdict(model_args),
        "eval_args": dataclasses.asdict(eval_args),
        "prune_args": dataclasses.asdict(prune_args),
        "cluster_args": dataclasses.asdict(cluster_args),
    }

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = convert_paths_to_str(all_args)

    output_path = pruned_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"All arguments saved to {output_path}")


def prune(
    observer_data,
    model,
    tokenizer,
    reap_args,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
):
    """
    Prune the model based on the observer data and clustering.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            # Calculate expert probabilities if not already present
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(observer_data, include_last_layers=prune_args.perserve_outliers)
        metrics = [
            "expert_proba",
            "ean_sum",
            "ean_mean",
            "weighted_expert_frequency_sum",
            "weighted_ean_sum",
            "reap",
            "reap_l2",
            "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if prune_args.prune_method == "ean_ca":
            ean = torch.zeros(num_experts, device=model.device, dtype=torch.float32)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i], dim=-1
                ).sum()
            _, experts_to_prune = torch.topk(ean, n_experts_to_prune, largest=False)
        else:
            prune_method = prune_args.prune_method
            if prune_method == "frequency":
                prune_method = "expert_frequency"
            saliency_data = observer_data[layer].get(prune_method)
            if saliency_data is None:
                raise ValueError(
                    f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                    f"Available keys: {list(observer_data[layer].keys())}"
                )
            _, experts_to_prune = torch.topk(
                saliency_data, n_experts_to_prune, largest=False
            )

        retained_expert_indicies = [
            i for i in range(num_experts) if i not in experts_to_prune
        ]
        # prune experts
        moe = get_moe(model, layer)
        if not model_attrs["fused"]:
            all_experts = getattr(moe, model_attrs["experts"])
            retained_experts = [all_experts[i] for i in retained_expert_indicies]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)
            if model.__class__.__name__.lower() == "Ernie4_5_MoEForCausalLM".lower():
                # transformers version >=4.54
                # prune expert score correction bias too
                moe.moe_statics.e_score_correction_bias.data = (
                    moe.moe_statics.e_score_correction_bias.data[
                        :, retained_expert_indicies
                    ]
                )

            # prune router
            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retained_expert_indicies, :]
            if getattr(router, "bias", None):
                router.bias.data = router.bias.data[retained_expert_indicies]
            router.out_features = len(retained_expert_indicies)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indicies]
                )
            setattr(moe, model_attrs["router"], router)
        else:
            # prune fused experts, only tested for llama-4
            moe.experts.gate_up_proj.data = moe.experts.gate_up_proj[
                retained_expert_indicies
            ]
            moe.experts.down_proj.data = moe.experts.down_proj[retained_expert_indicies]
            moe.num_experts = len(retained_expert_indicies)
            moe.router.weight.data = moe.router.weight.data[retained_expert_indicies]
            moe.router.out_features = len(retained_expert_indicies)
            if hasattr(moe.router, "num_experts"):  # transformers >= 4.54+
                moe.router.num_experts = len(retained_expert_indicies)

    # patch config and dump
    logger.info("Saving pruned model...")
    retained_experts = len(retained_expert_indicies)
    setattr(model.config, model_attrs["num_experts"], retained_experts)
    if model.__class__.__name__ == "Ernie4_5_MoeForCausalLM":  # remote-code verson
        model.config.moe_capacity = [
            retained_experts,
            retained_experts,
            retained_experts,
        ]

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model.save_pretrained(pruned_model_dir)
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    return pruned_model_dir


def get_pruned_model_dir(
    results_dir,
    n_experts_to_prune: str,
    total_experts: int,
    prune_args,
    seed: int,
) -> pathlib.Path:
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
    pruned_model_name = f"{prune_args.prune_method}"
    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"
    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")
    return pruned_model_dir


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
        )
    )
    reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args = (
        parser.parse_args_into_dataclasses()
    )
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError("Only one of perserve_super_experts or perserve_outliers can be set to True.")
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    # pruning
    logger.info("Start of pruning")
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set for pruning."
            )
        else:
            # Calculate n_experts_to_prune from compression_ratio
            total_experts = len(
                observer_data[next(iter(observer_data))]["expert_frequency"]
            )
            n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
            logger.info(
                f"Calculated n_experts to prune: {n_experts_to_prune} from compression_ratio: {cluster_args.compression_ratio}"
            )

    pruned_model_dir = get_pruned_model_dir(
        results_dir, n_experts_to_prune, total_experts, prune_args, reap_args.seed
    )
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
            "Skipping pruning step."
        )
    else:
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune(
            observer_data,
            model,
            tokenizer,
            reap_args,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )
        logger.info("pruning completed.")

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        tokenizer.save_pretrained(pruned_model_dir)
        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = pruned_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {pruned_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

        logger.info("Pruning completed.")

        dump_args_to_yaml(
            pruned_model_dir,
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
        )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = pruned_model_dir
        run_evaluate(model_args, pruned_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
