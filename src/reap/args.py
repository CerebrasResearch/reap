from dataclasses import dataclass, field
import dotenv
import os

dotenv.load_dotenv()


@dataclass
class ReapArgs:
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})  # 11, 99
    debug: bool = field(
        default=False, metadata={"help": "Enable debug mode for more verbose output."}
    )
    profile: bool = field(
        default=True, metadata={"help": "Enable profiling prior to run to avoid OOM."}
    )
    run_observer_only: bool = field(
        default=False,
        metadata={"help": "Whether to only run the observer to collect activation data."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation after merging experts."},
    )
    plot_clusters: bool = field(
        default=True,
        metadata={
            "help": "Whether to plot clusters after clustering experts. "
        }
    )
    smoke_test: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to run a smoke test on the merged model to ensure it works "
                "as expected prior to saving"
            )
        },
    )


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen/Qwen3-30B-A3B",
        metadata={
            "help": "Name of the model to use.",
            # "choices": [
            #     "mistralai/Mixtral-8x7B-Instruct-v0.1",
            #     "Qwen/Qwen3-30B-A3B",
            #     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            #     "baidu/ERNIE-4.5-21B-A3B-PT",
            #     "deepseek-ai/DeepSeek-V2-Lite-Chat",
            #     "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            #     "Qwen/Qwen3-30B-A3B-Instruct-2507",
            #     "openai/gpt-oss-20b",
            #     "openai/gpt-oss-120b",
            #     "zai-org/GLM-4.5-Air"
            # ],
        },
    )
    num_experts_per_tok_override: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override the number of experts per token. If None, uses the model's "
                "default number of experts per token."
            )
        },
    )


@dataclass
class DatasetArgs:
    dataset_name: str = field(
        default="theblackcat102/evol-codealpaca-v1",
        metadata={
            "help": (
                "Name of the dataset to use. Can be a single HuggingFace dataset name "
                "(e.g., 'theblackcat102/evol-codealpaca-v1') or a composite specification "
                "with comma-separated entries of <dataset>[<subset>](<split>):<num_samples>. "
                "Example: 'theblackcat102/evol-codealpaca-v1:4096,"
                "open-r1/Mixture-of-Thoughts[code]:4096,"
                "SWE-bench/SWE-smith-trajectories(tool):4096'. "
                "Use 'combined' to load pre-recorded combined observation data."
            ),            
        },
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Configuration name of the dataset."}
    )
    split: str = field(default="train", metadata={"help": "Dataset split to use."})
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the dataset."}
    )
    # for SFT only
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})


@dataclass
class ObserverArgs:
    samples_per_category: int = 1024
    split_by_category: bool = False
    select_only_categories: list[str] | str | None = field(
        default=None,
        metadata={
            "help": (
                "List of categories to select for observation. If None, all categories "
                "are selected."
            )
        },
    )
    batch_size: int = 1
    model_max_length: int | None = 2048
    return_vllm_tokens_prompt: bool = False
    truncate: bool = False
    overwrite_observations: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite existing observer data files."},
    )
    distance_measure: str = field(
        default="angular",
        metadata={
            "help": "Distance function to use for clustering.",
            "choices": ["angular", "euclidean", "jsd", "cka", "cosine"],
        },
    )
    output_file_name: str = field(
        default=f"observations_1024_cosine.pt",
        metadata={"help": "Name of the output file for observer data."},
    )
    record_pruning_metrics_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only record pruning metrics during observation to reduce "
                "memory usage and wall-clock time."
            )
        },
    )
    renormalize_router_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to renormalize topk router weights to sum to 1 if the model.config.norm_topk_prob is True."
            )
        }, 
    )

@dataclass
class ClusterArgs:
    cluster_description: str | None = field(
        default=None,
        metadata={
            "help": (
                "Description of the clustering run, used for naming output dir. "
                "If None, use: "
                "f'{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
                "{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}_"
                "softmax-{cluster_args.softmax_temperature}'"
            )
        },
    )
    expert_sim: str = field(
        default="ttm",
        metadata={
            "help": "Expert similiarty method.",
            "choices": [
                "ttm",
                "dynamic_ttm",
                "characteristic_activation",
                "routed_characteristic_activation",
                "router_logits",
                "online_characteristic_activation_dist"
            ],
        },
    )
    compression_ratio: float | None = field(
        default=0.5,
        metadata={
            "help": (
                "Compression ratio for clustering experts. If None, num_clusters must "
                "be set."
            )
        },
    )
    num_clusters: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of clusters to place experts into per layer. If None, "
                "num_clusters is calculated as int(num_experts * compression_ratio)."
            )
        },
    )
    cluster_method: str = field(
        default="agglomerative",
        metadata={
            "help": "Clustering method to use.",
            "choices": ["agglomerative", "kmeans", "spectral", "mc_smoe"],
        },
    )
    linkage_method: str = field(
        default="average",
        metadata={
            "help": "Linkage method for agglomerative clustering.",
            "choices": ["ward", "complete", "average", "single"],
        },
    )
    frequency_penalty: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to apply frequency penalty to expert similarity scores. "
                "If True, the frequency of each expert is used to scale the similarity"
            )
        },
    )
    softmax_temperature: float | None = field(
        default=None,
        metadata={
            "help": (
                "Temperature for softmax scaling of expert probabilities to calculate "
                "distance penalty vector. If 0 or None, expert probabilites are max "
                "normalized."
            )
        },
    )
    multi_layer: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of layers to merge at once. If None, merges all layers "
                "separately."
            )
        },
    )
    max_cluster_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "If not None, maximum number of experts per cluster. Only agglomerative"
                " cluster method supported"
            )
        }
    )
    singleton_super_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep super experts in a singleton when clustering"
            )
        }
    )
    singleton_outlier_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep outlier experts in a singleton when clustering"
            )
        }
    )



@dataclass
class MergeArgs:
    overwrite_merged_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite existing merged model files."
        },
    )
    merged_model_dir_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name of the merged model. If None, uses a concatenation of releveant hyperparameters:"
                "'merge_args.merge_method-merge_args.dom_as_base-merge_args.select_top_k-permute_merge_args.permute'"
            )
        },
    )
    merge_method: str = field(
        default="frequency_weighted_average",
        metadata={
            "help": "Method to use for merging experts.",
            "choices": [
                "frequency_weighted_average",
                "average",
                "ties",
                "multislerp",
                "sce",
                "karcher",
                "submoe"
            ],
        },
    )
    skip_first: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to skip the first layer when merging experts. "
            )
        }
    )
    skip_last: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to skip the last layer when merging experts. "
            )
        }
    )
    dom_as_base: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the most frequent expert as the base model for "
                "multislerp."
            )
        },
    )
    select_top_k: float = field(
        default=0.1,
        metadata={
            "help": (
                "Top-k percentage of weights to keep in non-dom experts for TIES."
            )
        }
    )
    permute: str | None = field(
        default=None,
        metadata={
            "help": (
                "Permutation to apply prior to merge"
            ),
            "choices": [
                None,
                "direct",
                "wm",
            ]
        },
    )
    save_as_tied_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the merged model as tied parameters. "
                "If False, saves merged experts as copies."
            )
        },
    )
    save_as_merged_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the merged model as a custom merged model. "
                "If False, saves merged experts as separate experts."
            )
        },
    )

@dataclass
class KdArgs:
    pass


@dataclass
class EvalArgs:
    use_server: bool = field(
        default=True,
        metadata={
            "help": "Whether to use a vllm server for evaluation. If False, uses hf backends."
        },
    )
    greedy: bool = field(
        default=True,
        metadata={
            "help": "Whether to use greedy decoding for evaluation. If False, uses sampling."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Temperature for sampling during evaluation. Ignored if greedy=True."
        },
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Top-p value for nucleus sampling during evaluation. Ignored if greedy=True."},
    )
    top_k: int = field(
        default=20,
        metadata={
            "help": "Top-k value for top-k sampling during evaluation. Ignored if greedy=True."
        },
    )
    min_p: float = field(
        default=0.00,
        metadata={
            "help": "Minimum probability for sampling during evaluation. Ignored if greedy=True."
        },
    )
    results_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Directory to save evaluation results. If None, results are saved "
                "in artifacts/model_name directory."
            )
        },
    )
    run_lm_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the merged model."},
    )
    run_evalplus: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation using evalplus."},
    )
    run_livecodebench: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation using livecodebench."},
    )
    run_wildbench: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation using wildbench."},
    )
    run_math: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation using math tasks."},
    )

    lm_eval_tasks: list[str] = field(
        default_factory=lambda: [
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "mmlu",
            "openbookqa",
            "rte",
        ],
        metadata={
            "help": "List of tasks to evaluate on using lm-eval.",
        },
    )
    evalplus_tasks: list[str] = field(
        default_factory=lambda: [
            "mbpp",
            "humaneval",
        ],
        metadata={
            "help": "List of tasks to evaluate on using evalplus.",
        },
    )
    server_log_file_name: str = field(
        default="server.log"
        if os.environ.get("SERVER_LOG_FILE_NAME") is None
        else os.environ["SERVER_LOG_FILE_NAME"],
        metadata={
            "help": "Name of the log file for the evaluation server.",
        },
    )
    vllm_port: int = field(
        default=8000,
        metadata={
            "help": "Port number for vLLM serve"
        },
    )
    parallel_tasks: int = field(
        default=32,
        metadata={
            "help": "Number of parallel tasks to run during evalplus evaluation."
        },
    )

@dataclass
class PruneArgs:
    overwrite_pruned_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite existing pruned model files."
        },
    )
    prune_method: str = field(
        default="frequency",
        metadata={
            "help": "Method to use for pruning experts.",
            "choices": [
                "frequency",
                "ean_ca",
                "ean_sum",
                'ean_mean',
                "weighted_frequency_sum",
                "weighted_ean_sum",
                "weighted_ean_sum_l2",
                "reap",
                "reap_l2",
                "max_activations"
            ]
        },
    )
    n_experts_to_prune: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of experts to keep after pruning. If None, use "
                "--compression-ratio."
            )
        },
    )
    perserve_super_experts: bool = field(
        default=False,
        metadata={
            "help": (
                r"Whether to perserve super experts when pruning. Excludes last 25% of layers"
            )
        }
    )
    perserve_outliers: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to perserve outlier experts when pruning, includes all layers"
            )
        }
    )

@dataclass
class QuantizationArgs:
    quantization_method: str = field(
        default="awq",
        metadata={
            "help": "Method to use for quantization.",
            "choices": ["awq","gptq"]
        },
    )
    scheme: str = field(
        default="W4A16",
        metadata={
            "help": "Quantization scheme to use.",
            "choices": ["W4A16",]
        },
    )
    group_size: int = field(
        default=128,
        metadata={
            "help": "Group / block size for quantization.",
        },
    )


@dataclass
class FSDPArgs:
    compute_environment: str = field(
        default="LOCAL_MACHINE", metadata={"help": "Compute environment type."}
    )
    debug: bool = field(default=False, metadata={"help": "Enable debug mode."})
    distributed_type: str = field(
        default="FSDP",
        metadata={
            "help": "Distributed training type.",
            "choices": [
                "NO",
                "MULTI_CPU",
                "MULTI_GPU",
                "MULTI_NPU",
                "FSDP",
                "DEEPSPEED",
            ],
        },
    )
    downcast_bf16: str = field(
        default="no",
        metadata={
            "help": "Whether to downcast bf16 operations.",
            "choices": ["no", "yes"],
        },
    )
    fsdp_auto_wrap_policy: str = field(
        default="TRANSFORMER_BASED_WRAP",
        metadata={
            "help": "FSDP auto wrap policy.",
            "choices": ["NO_WRAP", "SIZE_BASED_WRAP", "TRANSFORMER_BASED_WRAP"],
        },
    )
    fsdp_backward_prefetch_policy: str = field(
        default="BACKWARD_PRE",
        metadata={
            "help": "FSDP backward prefetch policy.",
            "choices": ["NO_PREFETCH", "BACKWARD_PRE", "BACKWARD_POST"],
        },
    )
    fsdp_forward_prefetch: bool = field(
        default=False, metadata={"help": "Enable FSDP forward prefetch."}
    )
    fsdp_cpu_ram_efficient_loading: bool = field(
        default=True, metadata={"help": "Enable CPU RAM efficient loading for FSDP."}
    )
    fsdp_offload_params: bool = field(
        default=False, metadata={"help": "Offload parameters to CPU with FSDP."}
    )
    fsdp_sharding_strategy: str = field(
        default="FULL_SHARD",
        metadata={
            "help": "FSDP sharding strategy.",
            "choices": ["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD", "HYBRID_SHARD"],
        },
    )
    fsdp_state_dict_type: str = field(
        default="SHARDED_STATE_DICT",
        metadata={
            "help": "FSDP state dict type.",
            "choices": ["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"],
        },
    )
    fsdp_sync_module_states: bool = field(
        default=True, metadata={"help": "Synchronize module states in FSDP."}
    )
    fsdp_transformer_layer_cls_to_wrap: str = field(
        default="BertLayer",
        metadata={"help": "Transformer layer class name to wrap with FSDP."},
    )
    fsdp_use_orig_params: bool = field(
        default=True, metadata={"help": "Use original parameters in FSDP."}
    )
    machine_rank: int = field(
        default=0, metadata={"help": "Rank of the machine in multi-machine setup."}
    )
    main_training_function: str = field(
        default="main", metadata={"help": "Name of the main training function."}
    )
    mixed_precision: str = field(
        default="bf16",
        metadata={
            "help": "Mixed precision mode for training.",
            "choices": ["no", "fp16", "bf16"],
        },
    )
    num_machines: int = field(
        default=1, metadata={"help": "Number of machines for distributed training."}
    )
    num_processes: int = field(
        default=2, metadata={"help": "Number of processes for distributed training."}
    )
    rdzv_backend: str = field(
        default="static",
        metadata={"help": "Rendezvous backend for distributed training."},
    )
    same_network: bool = field(
        default=True, metadata={"help": "Whether all machines are on the same network."}
    )
    use_cpu: bool = field(
        default=False, metadata={"help": "Force use of CPU instead of GPU."}
    )
