import os
import json
import torch
import copy
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- System Prompt Configuration ---
SYS_PROMPT = """You are an expert mathematician participating in the 2024 American Invitational Mathematics Examination (AIME). Your task is to solve complex mathematical problems by demonstrating rigorous logical reasoning and precise calculations.

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate Fisher Information Matrix for a specific model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the huggingface model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON data file")
    parser.add_argument("--output_dir", type=str, default="./Fisher_matrix", help="Directory to save the results")
    parser.add_argument("--output_filename", type=str, default="Fisher_matrix.pt", help="Filename for the saved tensor")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length for tokenizer")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- 0. Configure Paths and Directories ---
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, args.output_filename)

    # --- 1. Load JSON Samples ---
    with open(args.data_path, "r", encoding="utf-8") as f:
        reason_samples = json.load(f)
    print(f"Loaded {len(reason_samples)} samples from {args.data_path}.")

    # --- 2. Process the Model ---
    # List to store the final averaged Fisher_matrix (can be extended for multiple models if needed)
    final_fisher_matrices = []

    model_path = args.model_path
    
    print(f"\n=== Loading model: {model_path} ===")
    # Ensure dtype is appropriate for potential GPU use and calculations
    # Using bfloat16 if available for memory efficiency during forward/backward, but Fisher calc usually needs float32
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model and tokenizer loaded.")
    
    model.train() # Enable gradient calculation
    
    # Keep a copy of original weights to calculate the sensitivity (w - w*) or just use current weights w if deviation is not needed
    # Note: In the original logic, `original_value` seems to imply we multiply grad by the weight itself.
    original_state_dict = copy.deepcopy(model.state_dict())
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # List to store Fisher matrix for each sample for the current model
    model_specific_sample_fisher_matrices = [] 
    
    # --- For each input data sample ---
    for sample_idx, sample in enumerate(reason_samples):
        input_reannotated_assistant_content = sample.get("reannotated_assistant_content", "")
        input_problem = sample.get("problem", "")
        input_solution = sample.get("solution", "")
        
        # Constructing the input prompt
        input_prompt = (
            f"[INST] <<SYS>>\n{SYS_PROMPT}\n<</SYS>>\n\nProblem:\n{input_problem}\nModel Answer:\n{input_reannotated_assistant_content}\nSolution:\n{input_solution}\n"
            f"[/INST]"
        )
        
        inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = inputs["input_ids"].to(model.device)
        labels = input_ids.clone()
        
        model.zero_grad()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"Sample {sample_idx+1}/{len(reason_samples)}: Loss = {loss.item():.4f}")
        
        loss.backward()
        # print("Backward pass completed.")

        # Initialize Fisher matrix for the current sample (dimension [total_params])
        # Using float32 for squared values to prevent overflow/underflow
        current_sample_fisher_matrix = torch.zeros(total_params, dtype=torch.float32, device="cpu")
        
        idx_start = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # If grad is None (unused parameter in computation graph), skip or treat as zero
            if param.grad is None:
                # Still need to increment index to keep alignment
                idx_start += param.numel()
                continue

            grad_val = param.grad.detach() 
            original_value = original_state_dict[name].detach().to(param.device)
            
            # Calculate sensitivity value: gradient * original parameter value
            # This is a common approximation for importance: (g * w)^2
            sensitivity_val = grad_val * original_value 
            
            # Square each element for Fisher information approximation
            # Flatten and move to CPU to save GPU memory
            fisher_elements_for_param = (sensitivity_val ** 2).view(-1).cpu()
            
            numel = param.numel()
            
            # Safety check for index bounds
            if idx_start + numel > total_params:
                print(f"Error: Index out of bounds. Param: {name}, numel: {numel}, idx_start: {idx_start}, total: {total_params}")
                break 
            
            current_sample_fisher_matrix[idx_start : idx_start + numel] = fisher_elements_for_param
            idx_start += numel
        
        # Store the Fisher matrix for this specific sample
        model_specific_sample_fisher_matrices.append(current_sample_fisher_matrix)
        print(f"Processed Fisher matrix for sample {sample_idx+1}.")

    # After processing all samples, average their Fisher matrices
    if model_specific_sample_fisher_matrices:
        # Stack all sample Fisher matrices (list of [N] tensors -> [num_samples, N] tensor)
        stacked_sample_fishers = torch.stack(model_specific_sample_fisher_matrices)
        # Take the mean across the samples (dim=0)
        averaged_fisher_matrix_for_model = stacked_sample_fishers.mean(dim=0)
        final_fisher_matrices.append(averaged_fisher_matrix_for_model) # stored as list
        print(f"Averaged Fisher matrix computed successfully.")
    else:
        print(f"No samples were processed, no Fisher matrix generated.")
        
    print(f"Finished processing model.")
    
    # Cleanup memory
    del model, original_state_dict, model_specific_sample_fisher_matrices
    if 'stacked_sample_fishers' in locals(): del stacked_sample_fishers
    if 'averaged_fisher_matrix_for_model' in locals(): del averaged_fisher_matrix_for_model
    torch.cuda.empty_cache()

    # --- 3. Save Final Results ---
    if final_fisher_matrices:
        torch.save(final_fisher_matrices, save_file)
        print(f"\nFinal averaged Fisher matrices saved to {save_file}.")
    else:
        print("\nNo Fisher matrices were generated to save.")

if __name__ == "__main__":
    main()
    