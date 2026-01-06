import os
import json
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 0. Configure relevant paths and model names ---
model_names = {
    "Medical": "/data/transformers/Meditron3-Qwen2.5-7B"
}

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
'''

# JSON file location
json_path = "path to medical_samples.json"

# Directory and file path for saving sensitivity_matrix
save_dir = "path to sensitivity_matrix" # Modified directory name to reflect the new structure
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, "sensitivity_sparse_maps.pt") # Modified file name

# --- 1. Read JSON samples ---
with open(json_path, "r", encoding="utf-8") as f:
    reason_samples = json.load(f)
print(f"Loaded {len(reason_samples)} samples from {json_path}.")

# --- 2. Process all base_model ---
sensitivity_matrices = [] # Used to store a sparse sensitivity map of total_params length for each parameter

# Iterate through models
for model_key in ["Medical"]: # You can add other models as needed
    model_path = model_names[model_key]
    
    print(f"\n=== Loading model {model_key}: {model_path} ===")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model {model_key} and tokenizer loaded.")
    
    model.train() # Set to training mode for gradient calculation
    
    original_state_dict = copy.deepcopy(model.state_dict()) # Save initial model parameters
    
    # Count the total number of elements for all trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters for {model_key}: {total_params}")
    
    # --- For each input data ---
    for sample_idx, sample in enumerate(reason_samples):
        sent1 = sample.get("question", "")
        prompt = prompt_template.format(sent1=sent1)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        labels = input_ids.clone()
        
        model.zero_grad() # Clear gradients
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"[{model_key}] Sample {sample_idx+1}/{len(reason_samples)}: Loss = {loss.item():.4f}")
        
        loss.backward() # Backpropagate to compute gradients
        print(f"[{model_key}] Sample {sample_idx+1}: Gradients computed.")
        
        # current_trainable_param_idx_offset is used to locate the current trainable parameter's position in the total_params-length sparse map
        current_trainable_param_idx_offset = 0
        # Iterate through all named parameters of the model
        for name, param in model.named_parameters():
            if not param.requires_grad: # Only process trainable parameters
                continue

            # Get the number of elements for the current parameter
            numel_current_param = param.numel()

            # Create a zero tensor of length total_params to store the sensitivity of the current parameter, with other positions as 0
            # Note: Here, float32 type is chosen to store sensitivity values, as they are usually floating-point numbers
            param_specific_sparse_map = torch.zeros(total_params, dtype=torch.float32, device='cpu')
            
            grad_val = param.grad
            if grad_val is not None:
                original_value = original_state_dict[name].to(param.device) # Ensure device consistency
                # Calculate sensitivity value: gradient * original parameter value
                sensitivity_val_for_param = grad_val.detach() * original_value.detach()
                # Flatten the sensitivity_val of the current parameter into a 1D vector
                sensitivity_param_flat = sensitivity_val_for_param.view(-1).cpu()
                
                # Place the sensitivity_param_flat of the current parameter into the corresponding position in the sparse map
                param_specific_sparse_map[current_trainable_param_idx_offset : current_trainable_param_idx_offset + numel_current_param] = sensitivity_param_flat
            else:
                # If the gradient of a trainable parameter is None, it means it was not involved in this loss calculation.
                # Its sensitivity contribution will be 0, and the corresponding part in param_specific_sparse_map will remain 0.
                print(f"Warning: Gradient for trainable parameter {name} is None for sample {sample_idx+1}. Its sensitivity will be treated as zero in its sparse map.")

            # Append this total_params-length sparse sensitivity map, built for the current parameter, to the list.
            sensitivity_matrices.append(param_specific_sparse_map)
            # Now len(param_specific_sparse_map) should equal total_params

            # Update the starting position for the next trainable parameter in the total_params-length vector.
            current_trainable_param_idx_offset += numel_current_param

        # Checkpoint: Ensure that after all trainable parameters are iterated, the offset equals the total number of parameters.
        if current_trainable_param_idx_offset != total_params:
            print(f"Error: Mismatch in parameter count for sample {sample_idx+1}!")
            print(f"current_trainable_param_idx_offset ({current_trainable_param_idx_offset}) != total_params ({total_params})")

        print(f"[{model_key}] Sample {sample_idx+1}: Processed all parameters. `sensitivity_matrices` now has {len(sensitivity_matrices)} elements.")

    print(f"Finished processing model {model_key}.")
    
    del model
    del original_state_dict # Free up memory occupied by the original parameter dictionary
    torch.cuda.empty_cache()

# --- 3. Save sensitivity_matrix results for all models ---
print(f"\nAttempting to save {len(sensitivity_matrices)} sparse sensitivity maps...")
if sensitivity_matrices:
    print(f"Each map has a length of {len(sensitivity_matrices[0])} (should be {total_params}).")
    # Warning again: This will save a potentially very large file!
    # torch.save(sensitivity_matrices, save_file)
    # print(f"Sparse sensitivity maps saved to {save_file}.")
    print(f"Saving is commented out due to potentially very large file size. Please uncomment if you are sure.")
else:
    print("No sensitivity maps were generated to save.")