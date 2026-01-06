#!/bin/bash

# --- Configuration Section ---

# Input and Output Directories
# [PRIVACY PROTECTION]: Paths have been generalized. Please update with your actual paths.
INPUT_DIR="/path/to/your/input/directory"
OUTPUT_DIR="/path/to/your/output/directory"

# API Configuration
# [PRIVACY PROTECTION]: Replace with your actual API key.
API_KEY="sk-YOUR_REAL_API_KEY_HERE"
BASE_URL=""
MODEL_NAME="gpt-3.5-turbo-ca"

# Threading and Performance Hyperparameters
MAX_CONCURRENT_API_CALLS=100    # Maximum number of concurrent API calls
MAX_FILE_PROCESSING_THREADS=1   # Maximum number of threads processing files simultaneously
API_CALL_MAX_RETRIES=100        # Maximum retries for a single API call
API_CALL_BASE_DELAY_SECONDS=2   # Base delay for exponential backoff (in seconds)
API_CALL_TIMEOUT_SECONDS=30     # Timeout for a single API request (in seconds)
INTER_API_CALL_DELAY_SECONDS=0.2 # Fixed delay between API calls (managed by semaphore)

# --- Execution Section ---

echo "Starting Gibberish Detector..."
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Check if python3 is available, otherwise try python
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD gibberish_detector.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --max_concurrent_api_calls "$MAX_CONCURRENT_API_CALLS" \
    --max_file_processing_threads "$MAX_FILE_PROCESSING_THREADS" \
    --api_call_max_retries "$API_CALL_MAX_RETRIES" \
    --api_call_base_delay "$API_CALL_BASE_DELAY_SECONDS" \
    --api_call_timeout "$API_CALL_TIMEOUT_SECONDS" \
    --inter_api_call_delay "$INTER_API_CALL_DELAY_SECONDS"