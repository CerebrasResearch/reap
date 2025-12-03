#!/bin/bash

model_name=${1}
dataset_name=${2}
compression_ratio=${3}
server_log_file_name=${4}
port=${5}
seed=${6:-42}
# qa
run_lm_eval=${7:-true}
# coding
run_evalplus=${8:-true}
run_livecodebench=${9:-true}
# math
run_math=${10:-false}
# wildbench
run_wildbench=${11:-false}
singleton_super_experts=${12:-false}
singleton_outlier_experts=${13:-false}

echo "Running M-SMoE: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio, seed ${seed}, super expert singletons: ${singleton_super_experts}, outlier expert singletons: ${singleton_outlier_experts}"


cluster_description="m_smoe"
output_file_name="observations_1024_cosine-seed_${seed}.pt"
if [[ "${singleton_super_experts}" == "true" ]]; then
    merged_model_dir_name="m_smoe-super_expert_singletons-seed_${seed}_${compression_ratio}"
elif [[ "${singleton_outlier_experts}" == "true" ]]; then
    merged_model_dir_name="m_smoe-outlier_expert_singletons-seed_${seed}_${compression_ratio}"
else
    merged_model_dir_name="m_smoe-seed_${seed}_${compression_ratio}"
fi


# #mc-smoe
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
    --merged-model-dir-name ${merged_model_dir_name} \
    --cluster-description ${cluster_description} \
    --do-eval false \
    --seed ${seed} \
    --output-file-name ${output_file_name} \
    --singleton_super_experts ${singleton_super_experts} \
    --singleton_outlier_experts ${singleton_outlier_experts}

short_model_name=$(echo $model_name | cut -d'/' -f2)
short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)
merged_model="artifacts/${short_model_name}/${short_dataset_name}/non_uniform_merged_models/${merged_model_dir_name}/${cluster_description}"

bash experiments/eval.sh ${merged_model} ${seed} ${port} ${server_log_file_name} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${run_math} ${run_wildbench}

# echo "Removing safetensor files from ${merged_model}"
# rm ${merged_model}/*.safetensors

echo "Finished M-SMoE for model: ${model_name}, dataset: ${dataset_name}, compression ratio: ${compression_ratio}"

