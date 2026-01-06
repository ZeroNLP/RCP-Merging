#!/bin/bash

# ==============================================================================
# Configuration Script for Model Merging
# ==============================================================================

# --- Model Paths ---
# Path to the base DeepSeek model (Target for merging)
DEEPSEEK_PATH="/data/transformers/DeepSeek-R1-Distill-Qwen-7B/"

# Path to the Meditron model (Domain-specific model)
MEDITRON_PATH="/data/transformers/Meditron3-Qwen2.5-7B/"

# Path to the Qwen base model (Common ancestor/Base for subtraction)
QWEN_PATH="/data/transformers/Qwen2.5-7B/"

# --- Data & Matrix Paths ---
# Path to the pre-computed Fisher Information Matrix (.pt file)
FISHER_MATRIX_PATH="./Fisher_matrix/Fisher_matrix.pt"

# Path to the JSON file containing medical samples for sensitivity calculation
JSON_SAMPLES_PATH="./calibration/medical_sample/medical_samples.json"

# --- Output Configuration ---
# Directory where the merged model and config files will be saved
OUTPUT_DIR="OUTPUT_DIR"

# --- Hyperparameters ---
# Lambda value for the merging equation (Scaling factor for importance)
LAMBDA_VAL=1000.0

# ==============================================================================
# Execution
# ==============================================================================

echo "Starting model merging process..."
echo "DeepSeek Path: $DEEPSEEK_PATH"
echo "Lambda Value: $LAMBDA_VAL"

python merge_models.py \
    --deepseek_path "$DEEPSEEK_PATH" \
    --meditron_path "$MEDITRON_PATH" \
    --qwen_path "$QWEN_PATH" \
    --fisher_matrix_path "$FISHER_MATRIX_PATH" \
    --json_samples_path "$JSON_SAMPLES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lambda_val "$LAMBDA_VAL"

echo "Job finished."