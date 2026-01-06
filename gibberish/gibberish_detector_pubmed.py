import os
import json
import threading
import time
import random
import argparse
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
from concurrent.futures import ThreadPoolExecutor

# --- System Prompt for Gibberish Detection ---
SYSTEM_PROMPT = """You are an expert text analyzer. Your task is to determine if the provided text is **gibberish** or **meaningful**.

Classify the text as **gibberish** if it meets one or more of the following criteria:
1.  **Illogical mixing of languages:** The text contains a mix of Chinese and English (or other languages) in a nonsensical way that does not follow any logical grammatical structure.
2.  **Abuse of special characters:** The text is flooded with excessive special symbols, emojis, or control characters that disrupt or replace coherent content.
3.  **Senseless repetition:** The text repeats the same paragraphs, sentences, or phrases over and over without a clear purpose.
4.  **Meaningless character combinations:** The text consists of random character sequences or fragments that do not form recognizable words or convey any sense.
5.  **Suspected code or log snippets:** The text appears to be a fragment of programming code, a script, or system log information mixed into the main body of the content.

Otherwise, classify the text as **meaningful**. A text is meaningful if it can be understood as an attempt to convey information, even if it is short, grammatically incorrect, or factually wrong.

Respond with ONLY the word `gibberish` or `meaningful`. Do not add any explanations or other text."""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gibberish Detector using OpenAI API")
    
    # Path Arguments
    parser.add_argument("--input_dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed JSON files")
    
    # API Arguments
    parser.add_argument("--api_key", required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", default="https://api.chatanywhere.tech/v1", help="Base URL for the API")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-ca", help="Model name to use")
    
    # Performance Arguments
    parser.add_argument("--max_concurrent_api_calls", type=int, default=100, help="Max concurrent API calls")
    parser.add_argument("--max_file_processing_threads", type=int, default=1, help="Max file processing threads")
    parser.add_argument("--api_call_max_retries", type=int, default=100, help="Max retries per API call")
    parser.add_argument("--api_call_base_delay", type=float, default=2.0, help="Base delay for retries")
    parser.add_argument("--api_call_timeout", type=float, default=30.0, help="Timeout for API calls")
    parser.add_argument("--inter_api_call_delay", type=float, default=0.2, help="Delay between API calls")
    
    return parser.parse_args()

def is_gibberish(text_to_check, current_client, args, api_call_semaphore):
    """
    Calls OpenAI API to determine if text is gibberish, with retry and backoff.
    Returns the model's assessment ('gibberish', 'meaningful', or error string).
    """
    thread_id = threading.get_ident()
    if not text_to_check or not isinstance(text_to_check, str) or text_to_check.strip() == "":
        return "empty_or_invalid_input"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please analyze the following text: \"{text_to_check}\""}
    ]

    # Acquire semaphore: limits concurrent execution of this block
    with api_call_semaphore: 
        if args.inter_api_call_delay > 0:
            time.sleep(args.inter_api_call_delay)

        for attempt in range(args.api_call_max_retries):
            try:
                completion = current_client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=0.0,
                    timeout=args.api_call_timeout
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
                wait_time = args.api_call_base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"RateLimitError (Thread {thread_id}) for text '{text_to_check[:50]}...'. Retrying in {wait_time:.2f}s. (Attempt {attempt + 1}/{args.api_call_max_retries})")
                time.sleep(wait_time)
            except APIStatusError as e:
                if e.status_code == 429:
                    wait_time = args.api_call_base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"APIStatusError 429 (Thread {thread_id}) for text '{text_to_check[:50]}...'. Retrying in {wait_time:.2f}s. (Attempt {attempt + 1}/{args.api_call_max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"APIStatusError (Thread {thread_id}) for text '{text_to_check[:50]}...': Status {e.status_code}, Response: {e.response}")
                    return f"api_error_status_{e.status_code}"
            except APIConnectionError as e:
                print(f"APIConnectionError (Thread {thread_id}) for text '{text_to_check[:50]}...': {e}. Retrying (Attempt {attempt + 1}/{args.api_call_max_retries})")
                wait_time = args.api_call_base_delay * (2 ** attempt) / 2 + random.uniform(0, 0.5)
                time.sleep(wait_time)
            except Exception as e:
                error_str = str(e).upper()
                if "429" in error_str or "TOO_MANY_REQUESTS" in error_str or "CHATANYWHERE_ERROR" in error_str:
                    wait_time = args.api_call_base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Generic Error with 429 indicators (Thread {thread_id}) for text '{text_to_check[:50]}...': {e}. Retrying in {wait_time:.2f}s. (Attempt {attempt + 1}/{args.api_call_max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Unexpected Error (Thread {thread_id}) calling OpenAI API for text '{text_to_check[:50]}...': {e}")
                    return "api_error_unknown"

        print(f"Error (Thread {thread_id}): Max retries ({args.api_call_max_retries}) reached for text '{text_to_check[:50]}...'. Giving up.")
        return "api_error_max_retries"

def process_single_file(input_filepath, output_filepath, current_client, args, api_call_semaphore):
    """
    Processes a single JSON file. Designed to be run by ThreadPoolExecutor.
    Reads a JSON object, iterates through its key-value pairs, and processes 'model_answer' in nested objects.
    """
    filename = os.path.basename(input_filepath)
    thread_id = threading.get_ident()
    print(f"FileProcessor (Thread {thread_id}): Starting processing for file: {filename}...")

    output_data_list = []
    processed_items = 0
    failed_items = 0
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            json_data = json.load(f_in)

        if not isinstance(json_data, dict):
            print(f"Warning (FileProcessor Thread {thread_id}): Content of {filename} is not a dictionary (JSON object). Skipping.")
            return

        total_items = len(json_data)
        for key, nested_object in json_data.items():
            if not isinstance(nested_object, dict):
                print(f"Warning (FileProcessor Thread {thread_id}): Value for key '{key}' in {filename} is not a dictionary. Skipping.")
                continue

            text_to_check = nested_object.get("model_answer")

            if text_to_check is None:
                gibberish_assessment = "no_model_answer_found"
            else:
                gibberish_assessment = is_gibberish(text_to_check, current_client, args, api_call_semaphore)

            output_data_list.append({
                "key": key,
                "model_answer": text_to_check,
                "gibberish_assessment": gibberish_assessment
            })

            if "api_error" in gibberish_assessment or "unexpected" in gibberish_assessment:
                failed_items += 1
            processed_items += 1

            if processed_items % 10 == 0:
                 print(f"FileProcessor (Thread {thread_id}): Processed {processed_items}/{total_items} items in {filename} (Failures: {failed_items})")

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(output_data_list, f_out, indent=4, ensure_ascii=False)
        print(f"FileProcessor (Thread {thread_id}): Successfully processed {processed_items} items ({failed_items} failures) from {filename} and saved results to: {output_filepath}")

    except json.JSONDecodeError:
        print(f"Error (FileProcessor Thread {thread_id}): Could not decode JSON from {input_filepath}. Skipping.")
    except Exception as e:
        print(f"An unexpected error (FileProcessor Thread {thread_id}) occurred while processing {input_filepath}: {e}")

def main():
    args = parse_arguments()

    if "YOUR" in args.api_key:
        print("Error: Please replace the placeholder API key with your actual key in the Bash script.")
        return

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Client
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    # Initialize Semaphore based on args
    api_call_semaphore = threading.Semaphore(args.max_concurrent_api_calls)

    files_to_process = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    if not files_to_process:
        print("No JSON files found in the input directory.")
        return
        
    print(f"Found {len(files_to_process)} JSON files to process.")

    with ThreadPoolExecutor(max_workers=args.max_file_processing_threads) as executor:
        futures = []
        for filename in files_to_process:
            input_filepath = os.path.join(args.input_dir, filename)
            output_filepath = os.path.join(args.output_dir, filename)
            
            futures.append(executor.submit(
                process_single_file,
                input_filepath, output_filepath, client, args, api_call_semaphore
            ))

        print(f"\nMain: All {len(futures)} file processing tasks submitted. Waiting for completion...")
        
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Main: An error occurred in a file processing task {i+1}/{len(futures)}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n--- All file processing tasks have been processed. Total time: {end_time - start_time:.2f} seconds ---")