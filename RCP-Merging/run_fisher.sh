#!/bin/bash

# ==============================================================================
# Hyperparameter & Path Configuration
# ==============================================================================

# Path to the pre-trained HuggingFace model
# Example: "/data/transformers/DeepSeek-R1-Distill-Qwen-7B"
MODEL_PATH="/data/transformers/DeepSeek-R1-Distill-Qwen-7B"

# Path to the input dataset (JSON format) containing samples for calculation
DATA_PATH="./calibration/reason_sample/reason_sample.json"

# Directory where the output Fisher Matrix file will be saved
OUTPUT_DIR="./Fisher_matrix"

# Name of the output file
OUTPUT_FILENAME="Fisher_matrix.pt"

# Maximum sequence length for the tokenizer (truncation)
MAX_LENGTH=2048

# ==============================================================================
# Execution
# ==============================================================================

echo "Starting Fisher Matrix calculation..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"

python calculate_fisher.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_filename "$OUTPUT_FILENAME" \
    --max_length $MAX_LENGTH

if [ $? -eq 0 ]; then
    echo "Calculation completed successfully."
else
    echo "Calculation failed."
fi