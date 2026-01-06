#!/bin/bash

# --- File Paths Configuration (Anonymized) ---
# Replace these paths with your actual server paths
INPUT_DIR="/path/to/your/input/directory"
OUTPUT_DIR="/path/to/your/output/directory"

# --- API Configuration ---
# IMPORTANT: Use environment variables or a secure vault in production.
API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
BASE_URL=""
MODEL_NAME="gpt-3.5-turbo-ca"

# --- Data Structure Configuration ---
KEY_TYPE="problem"         # The key name used to identify each data item
TEXT_KEY="generated_text"  # The key name for the text content to be checked

# --- Performance Configuration ---
MAX_CONCURRENT_API_CALLS=100  # Max number of simultaneous API calls
MAX_FILE_PROCESSING_THREADS=1 # Max number of files to process concurrently
API_CALL_MAX_RETRIES=100
API_CALL_TIMEOUT_SECONDS=30

# --- System Prompt Configuration ---
# You can modify the definition of gibberish here
SYSTEM_PROMPT="You are an expert text analyzer. Your task is to determine if the provided text is **gibberish** or **meaningful**.

Classify the text as **gibberish** if it meets one or more of the following criteria:
1. **Illogical mixing of languages:** The text contains a mix of languages in a nonsensical way.
2. **Abuse of special characters:** Excessive special symbols or control characters.
3. **Senseless repetition:** Repetition of paragraphs or phrases without purpose.
4. **Meaningless character combinations:** Random character sequences.
5. **Suspected code or log snippets:** Fragments of code or system logs.

Otherwise, classify the text as **meaningful**. Respond with ONLY the word 'gibberish' or 'meaningful'."

# --- Execution ---
echo "Starting Gibberish Detection Pipeline..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python3 detect_gibberish.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL" \
  --model_name "$MODEL_NAME" \
  --key_type "$KEY_TYPE" \
  --text_key "$TEXT_KEY" \
  --max_concurrent_api_calls "$MAX_CONCURRENT_API_CALLS" \
  --max_file_threads "$MAX_FILE_PROCESSING_THREADS" \
  --max_retries "$API_CALL_MAX_RETRIES" \
  --timeout "$API_CALL_TIMEOUT_SECONDS" \
  --system_prompt "$SYSTEM_PROMPT"

echo "Pipeline finished."