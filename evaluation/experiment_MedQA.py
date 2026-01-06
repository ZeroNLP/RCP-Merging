import os
import json
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Configure paths
    model_name = "MODEL_NAME"
    merged_method = "Medical"
    device_name = 5
    reasoning = False
    data_dir = f"Medical/MedQA-USMLE-4-options-hf"
    if merged_method=="Medical":
        output_path = f"{OUTPUT_DIR}/MedQA/{model_name}.json"
        model_dir = f"{MODEL_PATH}/Medical/{model_name}"
    elif merged_method=="Basic":
        output_path = f"{OUTPUT_DIR}/Medical/MedQA/{model_name}.json"
        model_dir = f"{MODEL_PATH}/basic_merge/{model_name}"       
    files = glob.glob(os.path.join(data_dir, "test.jsonl")) + glob.glob(os.path.join(data_dir, "test.json"))
    if not files:
        raise FileNotFoundError(f"No JSON(.l) files found under {data_dir}")
    dataset_path = files[0]  # Take the first file

    # Load model and tokenizer
    device = torch.device(f"cuda:{device_name}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        # device_map="auto"
        device_map={"": f"cuda:{device_name}"}  # Force all layers to be assigned to CUDA:1
    )

    results = []
    prompt_template = '''You are a highly accurate medical question‐answering assistant.

You will receive:
  • input_text: A medical question stem describing a patient scenario or clinical problem.
  • ending0: The text of the first answer option.
  • ending1: The text of the second answer option.
  • ending2: The text of the third answer option.
  • ending3: The text of the fourth answer option.

Task:
  Select the single best option (ending0 through ending3) that correctly answers the question in input_text.

Output Requirement:
  • Respond with **only** the index of the chosen option: 0, 1, 2, or 3.
  • Do not include any additional words, punctuation, or explanation.

Question: {sent1}

ending0: {ending0}
ending1: {ending1}
ending2: {ending2}
ending3: {ending3}
'''

    # Read line by line and perform inference
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error at line {line_num}: {e}")
                continue  # Optional: skip the erroneous line, or raise e directly to terminate

            sent1 = item.get("sent1", "")
            endings = [item.get(f"ending{i}", "") for i in range(4)]
            label = item.get("label", None)

            # Format the prompt
            prompt = prompt_template.format(
                sent1=sent1,
                ending0=endings[0],
                ending1=endings[1],
                ending2=endings[2],
                ending3=endings[3]
            )

            # Model inference
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                if reasoning:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=32768,
                        temperature=0.6,
                        top_p=0.95,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id
                    ) 
                else:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id
                    )
            pred = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            # Collect results
            results.append({
                "sent1": sent1,
                "ending0": endings[0],
                "ending1": endings[1],
                "ending2": endings[2],
                "ending3": endings[3],
                "label": label,
                "prediction": pred
            })


    # Save to a JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()