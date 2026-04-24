#!/bin/bash
# =============================================================================
# VBVR-Multi-step Training: Wan2.2-I2V-A14B + Multi-step Data
# =============================================================================
# Train fresh LoRA on Wan2.2-I2V-A14B base model using 340K multi-step
# reasoning samples (34 task types).
#
# Prerequisites:
#   1. Multi-step data extracted to data/multistep/
#   2. Annotation JSONs generated via generate_multistep_annotations.py
#   3. Kill vLLM workers before running (they use all GPU memory)
#   4. Proxy must be set for HuggingFace model download
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_CONFIG_PATH="${REPO_DIR}/configs/multistep_dataset.json"

# Training parameters
HEIGHT=384
WIDTH=384
NUM_FRAMES=209
DATASET_REPEAT=1
LEARNING_RATE=1e-5
NUM_EPOCHS=1
LORA_RANK=32
SAVE_STEPS=500

# Output directories
HIGH_NOISE_OUTPUT_PATH="${REPO_DIR}/outputs/multistep/high_noise"
LOW_NOISE_OUTPUT_PATH="${REPO_DIR}/outputs/multistep/low_noise"

# Distributed training
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
NUM_PROCESSES=$((NUM_GPUS * NUM_NODES))

REMOVE_PREFIX_IN_CKPT="pipe.dit."

# Verify prerequisites
if [ ! -f "${DATASET_CONFIG_PATH}" ]; then
    echo "ERROR: ${DATASET_CONFIG_PATH} not found. Run generate_multistep_annotations.py first."
    exit 1
fi

# Set proxy for HuggingFace model download
export http_proxy='http://10.176.253.182:8080'
export https_proxy='http://10.176.253.182:8080'
export HTTP_PROXY='http://10.176.253.182:8080'
export HTTPS_PROXY='http://10.176.253.182:8080'
export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"
# HF_TOKEN must be set in environment before running

echo "=================================="
echo "VBVR-Multi-step Training"
echo "=================================="
echo "Model:           Wan-AI/Wan2.2-I2V-A14B (HuggingFace)"
echo "Dataset config:  ${DATASET_CONFIG_PATH}"
echo "Resolution:      ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "NUM_PROCESSES:   ${NUM_PROCESSES}"
echo "Learning Rate:   ${LEARNING_RATE}"
echo "LoRA Rank:       ${LORA_RANK}"
echo "Save every:      ${SAVE_STEPS} steps"
echo "=================================="

# ---- Train High Noise Model LoRA ----
echo "[Step 1/2] Training High Noise Model LoRA (timestep: 0 ~ 0.358)"

cd ${REPO_DIR}
mkdir -p ${HIGH_NOISE_OUTPUT_PATH}
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/wanvideo/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${HIGH_NOISE_OUTPUT_PATH} \
    --lora_base_model 'dit' \
    --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' \
    --lora_rank ${LORA_RANK} \
    --extra_inputs 'input_image' \
    --max_timestep_boundary 0.358 \
    --min_timestep_boundary 0 \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS}

echo "[Step 1/2] High Noise Model training complete."

# ---- Train Low Noise Model LoRA ----
echo "[Step 2/2] Training Low Noise Model LoRA (timestep: 0.358 ~ 1.0)"

mkdir -p ${LOW_NOISE_OUTPUT_PATH}
accelerate launch \
    --multi_gpu \
    --num_processes ${NUM_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --machine_rank ${NODE_RANK} \
    ${REPO_DIR}/examples/wanvideo/model_training/train.py \
    --dataset_config_path ${DATASET_CONFIG_PATH} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --remove_prefix_in_ckpt ${REMOVE_PREFIX_IN_CKPT} \
    --output_path ${LOW_NOISE_OUTPUT_PATH} \
    --lora_base_model 'dit' \
    --lora_target_modules 'q,k,v,o,ffn.0,ffn.2' \
    --lora_rank ${LORA_RANK} \
    --extra_inputs 'input_image' \
    --max_timestep_boundary 1 \
    --min_timestep_boundary 0.358 \
    --data_file_keys 'clip_path' \
    --save_steps ${SAVE_STEPS}

echo "[Step 2/2] Low Noise Model training complete."
echo "All training done!"
