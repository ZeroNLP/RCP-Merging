import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
use_reasoning = False
# Paths
model_name = "MODEL_NAME"
input_path = "/code_generation/test.jsonl"
output_path = f"{SAVE_PATH}Code/LiveCodeBench/{model_name}.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
device_name = 1
# Load dataset (JSONL)
samples = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        samples.append(json.loads(line))

# Load tokenizer and model
model_path = f"{MODEL_PATH}/two_models/Medical/{model_name}"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": f"cuda:{device_name}"}  # Force all layers to CUDA:1
)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)



# Base prompt
base_prompt = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the Python program\n\n"

# Inference
results = []
for item in samples:
    question_content = item.get('question_content', '')
    prompt = base_prompt + f"Question:\n{question_content}\n\n"
    print("question_content:")
    print(question_content)
    # Generate code
    generated_text = generate_answer(prompt)


    # Strip prompt from the returned text if included
    if generated_text.startswith(prompt):
        program = generated_text[len(prompt):]
    else:
        program = generated_text

    # Collect result
    result_entry = {
        'question_title': item.get('question_title'),
        'question_id': item.get('question_id'),
        'public_test_cases': item.get('public_test_cases'),
        'generated_program': program.strip()
    }
    print(result_entry)
    results.append(result_entry)

# Save results as JSON list
with open(output_path, 'w', encoding='utf-8') as out_f:
    json.dump(results, out_f, ensure_ascii=False, indent=2)