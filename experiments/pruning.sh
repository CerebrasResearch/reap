#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
model_name=${2:-"Qwen/Qwen3-30B-A3B"}
seed=${3:-42}

artifact_dir_name() {
    python - "$1" <<'PY'
import hashlib
import re
import sys

value = sys.argv[1]
if "," in value:
    print(f"composite_{hashlib.md5(value.encode()).hexdigest()[:8]}")
else:
    print(re.sub(r"[^\w\-_.]", "_", value.split("/")[-1]))
PY
}

server_log_file_name="pruning-${FIRST_DEVICE}-seed_${seed}.log"
run_lm_eval=true
run_evalplus=true
run_livecodebench=true

echo "Running pruning with model: $model_name on devices: $CUDA_VISIBLE_DEVICES"
echo "Logs will be saved to: $server_log_file_name"
echo "Evaluations: lm_eval: $run_lm_eval, evalplus: $run_evalplus, livecodebench: $run_livecodebench"
echo "Using seed: $seed"

datasets=(
    # "euclaise/WritingPrompts_curated"
    "theblackcat102/evol-codealpaca-v1"

)
compression_ratios=(
    0.50
    0.25
)
pruning_methods=(
    "frequency"
    "ean_sum"
    "ean_mean"
    "reap"
    "weighted_ean_sum"
)

output_file_name="observations_1024_cosine-seed_${seed}.pt"

for dataset_name in "${datasets[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
        for pruning_method in "${pruning_methods[@]}"; do
            echo "Running with model: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio, pruning method: $pruning_method"
            python src/reap/prune.py \
                --model-name $model_name \
                --dataset-name $dataset_name \
                --compression-ratio $compression_ratio \
                --prune-method $pruning_method \
                --profile false \
                --vllm_port $port \
                --server-log-file-name $server_log_file_name \
                --do-eval false \
                --distance_measure cosine \
                --seed $seed \
                --output_file_name ${output_file_name}

            short_model_name=$(artifact_dir_name "$model_name")
            short_dataset_name=$(artifact_dir_name "$dataset_name")
            model_dir="artifacts/${short_model_name}/${short_dataset_name}/pruned_models/${pruning_method}-seed_${seed}-${compression_ratio}"
            echo "evaluating model: ${model_dir}"
            bash eval.sh \
                $model_dir \
                $seed \
                $port \
                $server_log_file_name \
                $run_lm_eval \
                $run_evalplus \
                $run_livecodebench
            echo "Finished evaluating model: ${pruned_model}"

            echo "Removing safetensor files from ${model_dir}"
            rm ${model_dir}/*.safetensors
        done
    done
done
