import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configuration
device_name=0
model_name = "model_name"
reasoning = False # As specified, this will use the reasoning-specific generation parameters
model_path = f"{MODEL_PATH}/two_models/Finance/{model_name}"
input_json_path = "finance-tasks/ConviFinQA/test_conv.json"
# Ensure the save_path directory exists or the script has permission to create it.
save_path_dir = "{save_path}/results/two_models_Finance/Finance/ConvFinQA/"
save_path = os.path.join(save_path_dir, f"{model_name}.json")

# System prompt template. The model is expected to generate the content for {label}.
# So, we provide the prompt ending with "Output: "
system_prompt_template = """You are an economics-focused result generation machine. Your task is to analyze the provided "input" text and accurately answer the question presented at the end of the text. The answer should always be a floating-point number.

Input: {input_text}
Output: """

def main():
    # Check for CUDA availability and set device
    device = torch.device(f"cuda:{device_name}" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    # Load tokenizer and model
    print(f"INFO: Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        return

    print(f"INFO: Loading model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available, else float16
            device_map=device # Automatically distribute model across available devices
        )
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    # Ensure pad token is set for open-ended generation if not already set
    # This is crucial for some models and generation strategies
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print("INFO: tokenizer.pad_token_id is None. Setting to tokenizer.eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
        else:
            print("ERROR: tokenizer.pad_token_id and tokenizer.eos_token_id are None. Cannot set pad_token_id.")
            # Fallback or handle as per model's requirement if eos_token_id is also None
            # Forcing a default pad token if none exists (e.g. 0), but this might be model specific
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer))
            # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            # model.config.pad_token_id = tokenizer.pad_token_id
            print("INFO: Please ensure the model has a pad_token_id or eos_token_id for generation.")
            # return # Or proceed with caution

    # Load input data
    print(f"INFO: Loading data from {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {input_json_path}")
        return

    results = []

    print(f"INFO: Processing {len(test_data)} items...")
    for i, item in enumerate(test_data):
        input_text = item.get("input")

        if not isinstance(input_text, str): # Ensure input_text is a string
            print(f"WARNING: Item {i} has invalid or missing 'input' field (type: {type(input_text)}). Skipping.")
            results.append({**item, "generated_output_raw": None, "generated_output_float": None, "error": "Invalid or missing input field"})
            continue

        # Construct the prompt
        prompt = system_prompt_template.format(input_text=input_text)

        # Tokenize the input
        # No padding needed for single sequence generation, but truncation might be if prompt is too long
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - (32768 if reasoning else 2048)).to(model.device) # Ensure prompt doesn't exceed model limits minus generation space
        input_ids_length = inputs.input_ids.shape[1]


        print(f"\nINFO: Processing item {i+1}/{len(test_data)} (ID: {item.get('id', 'N/A')})")
        # print(f"  Prompt (start): {prompt[:300]}...") # Print a snippet of the prompt

        try:
            with torch.no_grad(): # Disable gradient calculations for inference
                if reasoning:
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask, # Important to pass attention_mask
                        max_new_tokens=32768, # As specified by user
                        temperature=0.6,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id # Use the configured pad_token_id
                    )
                else:
                    # This branch is for reasoning=False
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=4096, # Max total length (prompt + generation)
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

            # Decode the generated tokens.
            # outputs[0] contains the token IDs of the prompt + generation.
            # We only want the newly generated part.
            generated_ids = outputs[0][input_ids_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print(f"  Generated raw output: {generated_text}")

            parsed_float = None
            if generated_text: # Attempt parsing only if output is not empty
                try:
                    # Simple parsing: assumes the model outputs just the number or number first.
                    # For more complex outputs, regex or more advanced parsing might be needed.
                    # Example: extract first valid float found.
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", generated_text) # Basic float/integer regex
                    if match:
                        parsed_float = float(match.group(0))
                        print(f"  Parsed float from output: {parsed_float}")
                    else:
                        print(f"  WARNING: Could not find a number to parse as float in '{generated_text}'.")
                except ValueError:
                    print(f"  WARNING: Could not parse '{generated_text}' (or part of it) as float.")
            else:
                print("  WARNING: Model generated empty output.")


            current_result = {**item} # Start with original item data
            current_result["generated_output_raw"] = generated_text
            current_result["generated_output_float"] = parsed_float
            results.append(current_result)

        except Exception as e:
            print(f"ERROR: Error processing item {i} (ID: {item.get('id', 'N/A')}): {e}")
            results.append({
                **item,
                "generated_output_raw": None,
                "generated_output_float": None,
                "error": str(e)
            })
        
        # Optional: Clear CUDA cache if memory is an issue, especially with large models or batches
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Create directory for save_path if it doesn't exist
    print(f"\nINFO: Ensuring directory {save_path_dir} exists...")
    os.makedirs(save_path_dir, exist_ok=True)

    # Save results
    print(f"INFO: Saving results to {save_path}...")
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"INFO: Results successfully saved to {save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save results: {e}")

    print("\nINFO: Processing complete.")

if __name__ == "__main__":
    main()