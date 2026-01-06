import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import torch

# Configuration
device_name = 0
# model_name = "deepseek_qwen1.5b_code_math_task_arithmetic"
use_reasoning=False
model_name = "model_name"
test_dataset_path = "/code/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"
merged_method="Medical"
if merged_method == "Finance":
    model_path = f"{MODEL_PATH}/two_models/Finance/{model_name}"
    save_path = f"{save_path}/Code/HumanEval/{model_name}.json"
elif merged_method =="Basic":
    model_path =f"{MODEL_PATH}/basic_merge/{model_name}"
    save_path = f"{save_path}/Code/HumanEval/{model_name}.json"
elif merged_method =="Medical":
    model_path =f"{MODEL_PATH}/two_models/Medical/{model_name}"
    save_path = f"{save_path}/Code/HumanEval/{model_name}.json"
elif merged_method =="Llama":
    model_path =f"{MODEL_PATH}/Llama/{model_name}"
    save_path = f"{save_path}Code/HumanEval/{model_name}.json"

# Create directory for results
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Load dataset
df = pd.read_parquet(test_dataset_path)
device = torch.device(f"cuda:{device_name}")
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # device_map="auto"
    device_map={"": f"cuda:{device_name}"}  # Force all layers to be assigned to CUDA:1
)


tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if use_reasoning:
        outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=4096,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

# System prompt for 0-shot instruction
if merged_method=="Llama":
    system_prompt="""<s>[INST] <<SYS>>
You are an expert Python programmer. Your task is to complete the Python function based on the provided problem description and function signature.

Only output the raw code for the function's body. Do not include the function signature, the docstring, or any other explanatory text.
<</SYS>>

Complete the function below according to the problem description.
[/INST]"""
else:
    system_prompt = "You are an expert Python programmer. Complete the function below according to the problem description."

results = []
for idx, row in df.iterrows():
    prompt = row['prompt']
    
    # Manually concatenate Llama 2 Chat format
    full_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]"""
    
    generated_text = generate_answer(full_prompt)
    # Collect results
    result = row.to_dict()
    result = {
        "task_id": row["task_id"],
        "prompt": row["prompt"],
        "canonical_solution": row["canonical_solution"],
        "test": row["test"],
        "entry_point": row["entry_point"],
        "answer": generated_text  # Add the generated answer
    }
    print("generated_text")
    print(generated_text)
    results.append(result)

# Save results as JSON
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
