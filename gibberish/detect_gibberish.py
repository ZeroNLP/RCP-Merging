import os
import json
import threading
import time
import random
import argparse
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
from concurrent.futures import ThreadPoolExecutor

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect gibberish in JSON files using OpenAI API.")
    
    # Path Arguments
    parser.add_argument("--input_dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed JSON files")
    
    # API Arguments
    parser.add_argument("--api_key", required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", required=True, help="Base URL for the API")
    parser.add_argument("--model_name", required=True, help="Model name to use (e.g., gpt-3.5-turbo)")
    
    # Data Structure Arguments
    parser.add_argument("--key_type", required=True, help="Key name to identify each data item")
    parser.add_argument("--text_key", required=True, help="Key name for the text content to check")
    
    # Performance Arguments
    parser.add_argument("--max_concurrent_api_calls", type=int, default=100, help="Max simultaneous API calls")
    parser.add_argument("--max_file_threads", type=int, default=1, help="Max concurrent file processing threads")
    parser.add_argument("--max_retries", type=int, default=100, help="Max retries for API calls")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for a single API request")
    
    # Prompt Arguments
    parser.add_argument("--system_prompt", required=True, help="System prompt for the LLM")
    
    return parser.parse_args()

# --- Global Configuration (Populated via args) ---
args = parse_arguments()

API_CALL_BASE_DELAY_SECONDS = 2   # Initial delay for retries (exponential backoff)
INTER_API_CALL_DELAY_SECONDS = 0.2 # Small fixed delay between API calls

# --- OpenAI Client (Global) ---
client = OpenAI(
    api_key=args.api_key,
    base_url=args.base_url
)

# --- Semaphore for Limiting Concurrent API Calls ---
api_call_semaphore = threading.Semaphore(args.max_concurrent_api_calls)

def is_gibberish(text_to_check, current_client, model_name_for_api, system_prompt_for_api):
    """
    Calls the OpenAI API to determine if the text is gibberish, with retries and backoff.
    Returns the model's assessment ('gibberish', 'meaningful', or an error string).
    """
    thread_id = threading.get_ident()
    if not text_to_check or not isinstance(text_to_check, str) or text_to_check.strip() == "":
        return "empty_or_invalid_input"

    messages = [
        {"role": "system", "content": system_prompt_for_api},
        {"role": "user", "content": f"Please analyze the following text: \"{text_to_check}\""}
    ]

    with api_call_semaphore: # Acquire semaphore: limits concurrent execution of this block
        if INTER_API_CALL_DELAY_SECONDS > 0:
            time.sleep(INTER_API_CALL_DELAY_SECONDS)

        for attempt in range(args.max_retries):
            try:
                completion = current_client.chat.completions.create(
                    model=model_name_for_api,
                    messages=messages,
                    temperature=0.0,
                    timeout=args.timeout
                )
                response_content = completion.choices[0].message.content
                cleaned_response = response_content.strip().lower()

                if "gibberish" in cleaned_response:
                    return "gibberish"
                elif "meaningful" in cleaned_response:
                    return "meaningful"
                else:
                    print(f"Warning (Thread {thread_id}): Unexpected API response for text '{text_to_check[:50]}...': {response_content}")
                    return f"unexpected_api_response: {response_content}"

            except RateLimitError:
                wait_time = API_CALL_BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 1)
                print(f"RateLimitError (Thread {thread_id}). Retrying in {wait_time:.2f}s. (Attempt {attempt + 1}/{args.max_retries})")
                time.sleep(wait_time)
            except APIStatusError as e:
                if e.status_code == 429: 
                    wait_time = API_CALL_BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 1)
                    print(f"APIStatusError 429 (Thread {thread_id}). Retrying in {wait_time:.2f}s. (Attempt {attempt + 1}/{args.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"APIStatusError (Thread {thread_id}): Status {e.status_code}, Response: {e.response}")
                    return f"api_error_status_{e.status_code}"
            except APIConnectionError as e:
                print(f"APIConnectionError (Thread {thread_id}): {e}. Retrying (Attempt {attempt + 1}/{args.max_retries})")
                wait_time = API_CALL_BASE_DELAY_SECONDS * (2 ** attempt) / 2 + random.uniform(0, 0.5)
                time.sleep(wait_time)
            except Exception as e:
                error_str = str(e).upper()
                if "429" in error_str or "TOO_MANY_REQUESTS" in error_str:
                    wait_time = API_CALL_BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Generic Error with 429 indicators (Thread {thread_id}): {e}. Retrying in {wait_time:.2f}s.")
                    time.sleep(wait_time)
                else:
                    print(f"Unexpected Error (Thread {thread_id}): {e}")
                    return "api_error_unknown"

        print(f"Error (Thread {thread_id}): Max retries ({args.max_retries}) reached. Giving up.")
        return "api_error_max_retries"


def process_single_file(input_filepath, output_filepath, current_client, model_name_for_api, system_prompt_for_api):
    """
    Processes a single JSON file. This function is intended to be run by the ThreadPoolExecutor.
    """
    filename = os.path.basename(input_filepath)
    thread_id = threading.get_ident()
    print(f"FileProcessor (Thread {thread_id}): Starting processing for file: {filename}...")

    output_data_list = []
    processed_items = 0
    failed_items = 0
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            json_list = json.load(f_in)

        if not isinstance(json_list, list):
            print(f"Warning (FileProcessor Thread {thread_id}): Content of {filename} is not a list. Skipping.")
            return

        for index, item in enumerate(json_list):
            if not isinstance(item, dict):
                continue

            unique_id_val = item.get(args.key_type)
            prediction_text = item.get(args.text_key)

            if unique_id_val is None:
                print(f"Warning (FileProcessor Thread {thread_id}): '{args.key_type}' key missing in item {index} in {filename}. Skipping item.")
                continue
            
            if prediction_text is None:
                gibberish_assessment = "no_prediction_to_analyze"
            else:
                gibberish_assessment = is_gibberish(str(prediction_text), current_client, model_name_for_api, system_prompt_for_api)

            output_data_list.append({
                args.key_type: unique_id_val,
                "gibberish_assessment": gibberish_assessment
            })

            if "api_error" in gibberish_assessment or "unexpected" in gibberish_assessment :
                failed_items += 1
            processed_items +=1

            if processed_items % 10 == 0:
                 print(f"FileProcessor (Thread {thread_id}): Processed {processed_items}/{len(json_list)} items in {filename} (Failures: {failed_items})")

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(output_data_list, f_out, indent=4, ensure_ascii=False)
        print(f"FileProcessor (Thread {thread_id}): Successfully processed {processed_items} items ({failed_items} failures) from {filename} and saved results to: {output_filepath}")

    except json.JSONDecodeError:
        print(f"Error (FileProcessor Thread {thread_id}): Could not decode JSON from {input_filepath}. Skipping.")
    except Exception as e:
        print(f"An unexpected error (FileProcessor Thread {thread_id}) occurred while processing {input_filepath}: {e}")


def main_processor():
    """
    Processes all JSON files in the input directory using a ThreadPoolExecutor.
    """
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    files_to_process = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    if not files_to_process:
        print("No JSON files found in the input directory.")
        return
        
    print(f"Found {len(files_to_process)} JSON files to process.")

    with ThreadPoolExecutor(max_workers=args.max_file_threads) as executor:
        futures = []
        for filename in files_to_process:
            input_filepath = os.path.join(args.input_dir, filename)
            output_filepath = os.path.join(args.output_dir, filename)
            
            futures.append(executor.submit(
                process_single_file,
                input_filepath, output_filepath, client, args.model_name, args.system_prompt
            ))

        print(f"\nMain: All {len(futures)} file processing tasks submitted. Waiting for completion...")
        
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Main: An error occurred in a file processing task {i+1}/{len(futures)}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main_processor()
    end_time = time.time()
    print(f"\n--- All file processing tasks have been processed. Total time: {end_time - start_time:.2f} seconds ---")