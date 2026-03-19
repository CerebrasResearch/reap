#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
port=8000
WORKING_DIR=$(pwd)
method_description=${1:-"m_smoe"}  # hc_smoe, wmean
use_lora=${2:-true}  # true, false
lora_r=${3:-8}
lora_alpha=${4:-16}
model_name=${5:-"Qwen/Qwen3-30B-A3B"}
dataset_name=${6:-"theblackcat102/evol-codealpaca-v1"}
compression_ratio=${7:-0.50}

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

if [[ "${use_lora}" == "true" ]]; then
    server_log_file_name="sft-${method_description}-lora-r${lora_r}-a${lora_alpha}.log"
else
    server_log_file_name="sft-${method_description}-no-lora.log"
fi


echo "Running ${method_description} w/ lora=${use_lora}: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio"
echo "Lora r: ${lora_r}, Lora alpha: ${lora_alpha}"

if [[ "${method_description}" == "hc_smoe" ]]; then
    echo "Using HC-SMoE"
    # hc-smoe
    python src/reap/main.py \
        --compression_ratio ${compression_ratio} \
        --model-name ${model_name} \
        --dataset-name ${dataset_name} \
        --merge-method frequency_weighted_average \
        --profile false \
        --server_log_file_name $server_log_file_name \
        --vllm-port $port \
        --expert-sim characteristic_activation \
        --distance_measure euclidean \
        --linkage-method average \
        --frequency-penalty false \
        --skip-first false \
        --skip-last false \
        --merged-model-dir-name "hc_smoe-${compression_ratio}" \
        --cluster-description "hc_smoe" \
        --do-eval false \
        --save-as-tied-params false

    short_model_name=$(artifact_dir_name "$model_name")
    short_dataset_name=$(artifact_dir_name "$dataset_name")
    model_dir="artifacts/${short_model_name}/${short_dataset_name}/merged_models/hc_smoe-${compression_ratio}/hc_smoe"
elif [[ "${method_description}" == "m_smoe" ]]; then
    echo "Using M-SMoE"
    # m-smoe
    python src/reap/main.py \
        --compression-ratio ${compression_ratio} \
        --model-name ${model_name} \
        --dataset-name ${dataset_name} \
        --merge-method frequency_weighted_average \
        --permute wm \
        --cluster-method mc_smoe \
        --profile false \
        --server_log_file_name $server_log_file_name \
        --vllm-port $port \
        --expert-sim router_logits \
        --distance_measure cosine \
        --frequency-penalty false \
        --merged-model-dir-name "m_smoe-${compression_ratio}" \
        --cluster-description "m_smoe" \
        --do-eval false

        short_model_name=$(artifact_dir_name "$model_name")
        short_dataset_name=$(artifact_dir_name "$dataset_name")
        model_dir="artifacts/${short_model_name}/${short_dataset_name}/non_uniform_merged_models/m_smoe-${compression_ratio}/m_smoe"
elif [[ "${method_description}" == "wmean" ]]; then
    echo "Using WMEAN"
    python src/reap/prune.py \
        --model-name $model_name \
        --dataset-name $dataset_name \
        --compression-ratio $compression_ratio \
        --prune-method reap \
        --profile false \
        --vllm_port $port \
        --server-log-file-name $server_log_file_name \
        --do-eval false \
        --distance_measure cosine
    short_model_name=$(artifact_dir_name "$model_name")
    short_dataset_name=$(artifact_dir_name "$dataset_name")
    model_dir="artifacts/${short_model_name}/${short_dataset_name}/pruned_models/${pruning_method}-${compression_ratio}"
fi


if [[ "${use_lora}" == "true" ]]; then
    echo "Now doing SFT with LoRA on ${model_dir}"
    if [[ ${CUDA_VISIBLE_DEVICES} == *","* ]]; then
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
        echo "Detected multiple GPUs: ${num_gpus}"
        echo "Running lora with FSDP"
        # # Now SFT w/ lora 
        lora_r=8
        accelerate launch \
            --config_file config/fsdp_v2_4-proc.yaml \
            --num_processes 4 \
            src/reap/sft.py \
            --model_name_or_path ${model_dir} \
            --dataset_name ${dataset_name} \
            --assistant_only_loss true \
            --learning_rate 5.0e-5 \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.03 \
            --packing \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --max_length 2048 \
            --save_strategy steps \
            --save_steps 1000 \
            --attn_implementation flash_attention_2 \
            --torch_dtype bfloat16 \
            --save_total_limit 1 \
            --output_dir ${model_dir}/sft-lora-r_${lora_r} \
            --overwrite_output_dir \
            --run_name ${short_model_name}-${short_dataset_name}-${method_description}-${compression_ratio}-sft-lora-r_${lora_r} \
            --do_train \
            --do_eval \
            --eval_strategy steps \
            --eval_steps 1000 \
            --report_to wandb \
            --mixed_precision 'no' \
            --gradient_checkpointing false \
            --use_peft \
            --lora_r ${lora_r} \
            --lora_alpha ${lora_alpha} \
            --num_train_epochs 1
    else
        # lora single device
        python src/reap/sft.py \
            --model_name_or_path ${model_dir} \
            --dataset_name ${dataset_name} \
            --assistant_only_loss true \
            --learning_rate 5.0e-5 \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.03 \
            --packing \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --max_length 2048 \
            --save_strategy steps \
            --save_steps 1000 \
            --attn_implementation flash_attention_2 \
            --torch_dtype bfloat16 \
            --save_total_limit 1 \
            --output_dir ${model_dir}/sft-lora-r_${lora_r} \
            --overwrite_output_dir \
            --run_name ${short_model_name}-${short_dataset_name}-${method_description}-${compression_ratio}-sft-lora-r_${lora_r} \
            --do_train \
            --do_eval \
            --eval_strategy steps \
            --eval_steps 1000 \
            --report_to wandb \
            --mixed_precision 'no' \
            --gradient_checkpointing false \
            --use_peft \
            --lora_r ${lora_r} \
            --lora_alpha ${lora_alpha} \
            --num_train_epochs 1
    fi
    sft_dir="${model_dir}/sft-lora-r_${lora_r}"

# no lora
else
    echo "Now doing SFT without LoRA on ${model_dir}"
    accelerate launch \
        --config_file config/fsdp_v2_4-proc.yaml \
        src/reap/sft.py \
        --model_name_or_path ${model_dir} \
        --dataset_name ${dataset_name} \
        --assistant_only_loss true \
        --learning_rate 5.0e-5 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --packing \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_length 2048 \
        --save_strategy steps \
        --save_steps 1000 \
        --attn_implementation flash_attention_2 \
        --torch_dtype bfloat16 \
        --save_total_limit 1 \
        --output_dir ${model_dir}/sft \
        --overwrite_output_dir \
        --run_name ${short_model_name}-${short_dataset_name}-${method_description}-${compression_ratio}-sft \
        --do_train \
        --do_eval \
        --eval_strategy steps \
        --eval_steps 1000 \
        --report_to wandb \
        --mixed_precision 'no' \
        --gradient_checkpointing false \
        --num_train_epochs 1

    sft_dir="${model_dir}/sft"
fi

echo "Finished SFT, now evaluating ${sft_dir}"
bash eval.sh ${sft_dir} ${model_name} ${port} ${server_log_file_name}
