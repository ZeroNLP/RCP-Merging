import os
import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # Paths
    use_reasoning=False
    device_name=3
    model_name ="model_name"

    data_path = 'gpqa_diamond/data/test-00000-of-00001.parquet'

    model_dir = f'{MODEL_PATH}/two_models/Medical/{model_name}'
    output_path = f'{save_path}/results/two_models/QA/GPQA/{model_name}.json'

    # Load dataset
    df = pd.read_parquet(data_path)
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        # device_map="auto"
        device_map={"": f"cuda:{device_name}"}  # Force all layers to be allocated to CUDA:1
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Inference'):
        question_content = row.get('problem', '')
        prompt = "What is the correct answer to this question: "
        prompt += f"Question:{question_content}"

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        if use_reasoning:
            output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.90,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the generated portion after the prompt
        generated_text = generated[len(prompt):].strip()
        print(generated_text)
        # Collect fields
        result = {
            'problem': question_content,
            'domain': row.get('domain', None),
            'solution': row.get('solution', None),
            'generated_text': generated_text
        }
        results.append(result)

    # Save results as JSON list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()