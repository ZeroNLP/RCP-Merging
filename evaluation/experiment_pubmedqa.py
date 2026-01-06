import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from pathlib import Path
import json


use_auto = False
device_number=2
model_name = "MODEL_NAME"
reasoning = False
MODEL_PATH =f"{MODEL_PATH}/{model_name}"

DATA_PATH = "pubmed_qa/pqal_test.json"

SAVE_PATH = Path(f"{SAVE_PATH}/PubMedQA/{model_name}.json")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
# Load model and tokenizer
device = f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if use_auto:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{device_number}"
    )
model.eval()

def generate_answer(question, contexts):
    sys_prompt = """ [INST] <<SYS>> 
    You are the world's best diagnostic expert in the field of biology, here is a Question and the corresponding Context background. 
    Generate an answer from the model given a question and context. 
    Answer only with 'yes', 'no', or 'maybe'.
    <</SYS>>
    """
    prompt = f"System: {sys_prompt}\n\nQuestion: {question}\n\nAnswer: [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if reasoning:
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        output = model.generate(**inputs, max_new_tokens=4096, do_sample=False,temperature=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Load dataset
with open(DATA_PATH, "r") as f:
    dataset = json.load(f)
count =0

# Process dataset
for key, entry in dataset.items():
    count +=1
    question = entry["QUESTION"]
    contexts = entry["CONTEXTS"]
    model_answer = generate_answer(question, contexts)
    print(f"model_answer: {model_answer}, Count: {count}")
    dataset[key]["model_answer"] = model_answer

# Save output
with open(SAVE_PATH, "w") as f:
    json.dump(dataset, f, indent=4)
