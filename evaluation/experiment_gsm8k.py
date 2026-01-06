import re
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

def load_gsm8k(path):
    df = pd.read_parquet(path)
    return [{"question": row["question"], "answer": row["answer"]} for _, row in df.iterrows()]

train_data = load_gsm8k("math/gsm8k/main/train-00000-of-00001.parquet")
test_data = load_gsm8k("math/gsm8k/main/test-00000-of-00001.parquet")
device_name = 4
use_reasoning = False
device = torch.device(f"cuda:{device_name}")
merged_method = "Medical"
model_name = "model_name"

use_auto = False

# Load local model
if merged_method == "Finance":
    model_path =f"{MODEL_PATH}/two_models/Finance/{model_name}"
    save_path = f"{save_path}/results/two_models_Finance/Math/GSM8K/{model_name}.json"
elif merged_method =="Basic":
    model_path =f"{MODEL_PATH}/basic_merge/{model_name}"
    save_path = f"{save_path}/results/basic_merge/Math/GSM8K/{model_name}.json"
elif merged_method =="Medical":
    model_path =f"{MODEL_PATH}/two_models/Medical/{model_name}"
    save_path = f"{save_path}/results/two_models/Math/GSM8K/{model_name}.json"
elif merged_method =="Llama":
    model_path =f"{MODEL_PATH}/Llama/{model_name}"
    save_path = f"{save_path}/Llama/Math/GSM8K/{model_name}.json"
model_path="{MODEL_PATH}/two_models/Medical/EWC_only_penalty"
# model_path = f"/data/transformers/{model_name}/"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def create_prompt(shots, test_question):
    prompt = "Solve math problems with step-by-step reasoning and put the final answer in \\boxed{}.\n\n"
    for i, shot in enumerate(shots[:4], 1):
        prompt += f"Example {i}:\nQuestion: {shot['question']}\nAnswer: {shot['answer']}\n\n"
    prompt += f"Now solve this:\nQuestion: {test_question}\nAnswer:"
    return prompt

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if use_reasoning:
        outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.9,
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

def evaluate_and_save():

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    results = []
    for test_example in tqdm(test_data[:]):  # First 100 for testing
        prompt = create_prompt(train_data, test_example['question'])
        generated_text = generate_answer(prompt)
        print(generated_text)
        # Extract answers
        model_answer = re.search(r'\\boxed{([^}]+)}', generated_text)
        ground_truth = re.search(r'\\boxed{([^}]+)}', test_example['answer'])
        results.append({
            "question": test_example["question"],
            "reference_answer": test_example["answer"],
            "generated_answer": generated_text,
            "prediction": model_answer.group(1) if model_answer else None,
            "ground_truth": ground_truth.group(1) if ground_truth else None,
            "correct": (model_answer and ground_truth and 
                       model_answer.group(1) == ground_truth.group(1))
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['correct'])
    print(f"4-shot Accuracy: {correct/len(results)*100:.2f}%")

# Run evaluation
evaluate_and_save()