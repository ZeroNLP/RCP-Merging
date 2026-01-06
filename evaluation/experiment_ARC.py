from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
import numpy as np
import torch
# Specify the model folder path
device_name=2
model_name = "model_name"
model_type ="qwen"
reasoning = False
model_path = f"{MODEL_PATH}/two_models/Medical/EWC_only_penalty"
file_path = "ai2_arc/ARC-Challenge/test-00000-of-00001.parquet"
save_path = f"{save_path}/results/Llama/QA/ARC-C/{model_name}.json"
device = torch.device(f"cuda:{device_name}") 
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Load the model and automatically assign it to available GPUs
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)
print(model_name)
# Load data
df = pd.read_parquet(file_path)

# System prompt
if model_type=="qwen":
    sys_prompt = (
        "You are the best encyclopedia in the world, here is a paragraph of the question "
        "and four possible options for this question. You need to choose the correct option "
        "from the four ABCD options as your generated content."
    )
elif model_type =="llama":
    sys_prompt = """You are an advanced AI model with strong reasoning capabilities. You are tasked with answering challenging science questions from the ARC-Challenge dataset.
    These questions require careful analysis and understanding of scientific principles.
    Please read the question and the provided options thoroughly.
    Your final output should be *only* the capital letter (A, B, C, D, etc.) of the most correct option. Strive for accuracy."""

# Output data list
results = []

# Iterate over each row in the dataset
for idx, row in df.iterrows():
    input_text = row[1]  # The second column is the question description
    options = row[2]['text']  # The 'text' key in the third column stores the options
    if len(options) < 4:
        print(f"Row {idx} has fewer than 4 options, skipping.")
        continue

    # Extract options
    option_a = options[0]
    option_b = options[1]
    option_c = options[2]
    option_d = options[3]

    # Construct the input prompt
    input_prompt = (
        f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{input_text}\n"
        f"Four options are:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}[/INST]"
    )
    # print(input_prompt)
    # Tokenize the input text and convert it to a tensor
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

    # Generate model output
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
            max_new_tokens=4096, # Max total length (prompt + generation)
            pad_token_id=tokenizer.pad_token_id
            )

    # Decode the output
    generated_part = outputs[:, inputs.input_ids.shape[-1]:]  # Keep the 2D structure for slicing
    generated_text = tokenizer.decode(generated_part[0], skip_special_tokens=True)

    # Save the results
    result = row.to_dict()  # Convert the current row to a dictionary
    result['choices']['text'] = list(result['choices']['text'])  # Convert to list type
    result['choices']['label'] = list(result['choices']['label'])  # Convert to list type
    result['answer'] = generated_text  # Add the model-generated answer
    print("generated_text")
    print(generated_text)
    results.append(result)
    # print(result)

# Save the results to a JSON file
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {save_path}")