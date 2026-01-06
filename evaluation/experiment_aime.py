# /data/jyyang/model_merge/mergekit/datasets/math/aime_2024/data/train-00000-of-00001.parquet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
import numpy as np
# Specify model folder path
device_name=0
merged_method = "Medical"
model_name = "model_name"
reasoning = True
use_auto = False

if merged_method == "Medical":
    model_path = f"{MODEL_PATH}/two_models/Medical/{model_name}"
    save_path = f"{SAVE_PATH}/two_models/Math/AIME/{model_name}.json"
elif merged_method == "All":
    model_path =f"{MODEL_PATH}/main_experiment/{model_name}"
    save_path = f"{SAVE_PATH}/main_experiment/AIME/{model_name}.json"
elif merged_method == "solo":
    model_path =f"/data/transformers/{model_name}"
    save_path = f"{SAVE_PATH}/main_experiment/AIME/{model_name}.json"
elif merged_method == "Finance":
    model_path =f"/data/transformers/{model_name}"
    save_path = f"{SAVE_PATH}/Finance/Math/AIME/{model_name}.json"
elif merged_method =="Basic":
    model_path =f"{MODEL_PATH}/basic_merge/{model_name}"
    save_path = f"{SAVE_PATH}/basic_merge/Math/AIME/{model_name}.json"

file_path = "aime_2024/data/train-00000-of-00001.parquet"

# model_path = f"/data/transformers/{model_name}"

device = torch.device(f"cuda:{device_name}") 
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
# Load model and automatically assign to available GPU

if use_auto:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{device_name}"
    )


# Load data
df = pd.read_parquet(file_path)

# System prompt
sys_prompt = """You are an expert mathematician participating in the 2024 American Invitational Mathematics Examination (AIME). Your task is to solve complex mathematical problems by demonstrating rigorous logical reasoning and precise calculations.

Problem-Solving Guidelines:
1. ​**Problem Analysis**
   - Carefully read the problem statement twice
   - Identify known quantities and required outputs
   - Recognize mathematical domains involved (algebra, geometry, number theory, combinatorics, etc.)

2. ​**Solution Construction**
   - Break down the problem into logical steps
   - Apply appropriate theorems/formulas with justification
   - Maintain dimensional/unit consistency where applicable
   - Handle special constraints (e.g. integer solutions in [0,999])

3. ​**Verification**
   - Check intermediate results for arithmetic accuracy
   - Validate answer satisfies all problem conditions
   - Consider alternative approaches for cross-verification

Format Requirements:
1. Present final answer as: \boxed{XXX} (exactly 3 digits)
2. Express all fractions in simplest form
3. Use LaTeX for mathematical notation
4. Avoid explanatory text in final answer

Special Instructions:
- Prioritize elegant solutions over brute-force methods
- Explicitly state non-trivial inferences
- Handle edge cases with proper justification
- Remember AIME problems often have unique integer solutions between 0-999
"""

# Output data list
results = []
count =0
# Iterate through each row in the dataset
for idx, row in df.iterrows():
    input_text = row[1]  # Second column is the problem description
    print(f"input_text: {input_text}")
    # Construct input prompt
    input_prompt = (
        f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{input_text}\n"
        f"[/INST]"
    )
    # print(input_prompt)
    # Tokenize input text and convert to tensors
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

    # Generate model output
    if reasoning:
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(**inputs, max_length=4096, do_sample=False)

    # Decode output
    # generated_part = outputs[0][:, inputs.input_ids.shape[-1]:]  # Extract newly generated part
    generated_part = outputs[:, inputs.input_ids.shape[-1]:]  # Keep 2D structure for slicing

    generated_text = tokenizer.decode(generated_part[0], skip_special_tokens=True)
    # print(generated_text)
    # Save results
    result = row.to_dict()  # Convert current row to dictionary
    # result['choices']['text'] = list(result['choices']['text'])  # Convert to list type
    # result['choices']['label'] = list(result['choices']['label'])  # Convert to list type
    result['generated_text'] = generated_text  # Add model generated answer

    print("answer:")
    print(generated_text)
    results.append(result)
    count+=1
    print(f"Count: {count}")
    # print(result)

# Save results to JSON file
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {save_path}")