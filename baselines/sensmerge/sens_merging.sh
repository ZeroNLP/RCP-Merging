#!/bin/bash

# --- Configuration Section ---

# Paths to the models
# BASE_MODEL: Path to the base pre-trained model (e.g., Llama-3)
BASE_MODEL="/data/transformers/Meta-Llama-3.1-8B"

# MODEL1: Path to the first expert model (e.g., Medical domain)
MODEL1_PATH="/data/transformers/Llama3-OpenBioLLM-8B"

# MODEL2: Path to the second expert model (e.g., Reasoning domain)
MODEL2_PATH="/data/transformers/DeepSeek-R1-Distill-Llama-8B"

# Paths to the calibration datasets
# DATA_MEDICAL: JSON dataset for the medical model
DATA_MEDICAL="../../RCP-Merging/calibration/medical_sample/medical_samples.json"

# DATA_REASONING: JSON dataset for the reasoning model
DATA_REASONING="../../RCP-Merging/calibration/reason_sample/reason_sample.json"

# Output Configuration
# OUTPUT_DIR: Where the merged model will be saved
OUTPUT_DIR="./merged_output"

# Sens-Merging Hyperparameters
# NUM_SAMPLES: Number of samples 'm' used for sensitivity calculation
NUM_SAMPLES=100

# TEMP: Temperature 'T' for the softmax function in the final coefficient calculation
# Controls the sharpness of the weighting between models.
TEMP=1.0

# SYSTEM_PROMPT: System prompt specifically used for the reasoning dataset input construction
SYSTEM_PROMPT="You are a helpful assistant that solves problems."

# --- Execution Section ---

echo "Starting Sens-Merging process..."
echo "Base Model: $BASE_MODEL"
echo "Output Directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python3 sens_merging.py \
    --base_model_path "$BASE_MODEL" \
    --model1_path "$MODEL1_PATH" \
    --model2_path "$MODEL2_PATH" \
    --medical_dataset_path "$DATA_MEDICAL" \
    --reasoning_dataset_path "$DATA_REASONING" \
    --output_path "$OUTPUT_DIR" \
    --num_calibration_samples $NUM_SAMPLES \
    --softmax_temperature $TEMP \
    --sys_prompt_reasoning "$SYSTEM_PROMPT"

echo "Job finished."