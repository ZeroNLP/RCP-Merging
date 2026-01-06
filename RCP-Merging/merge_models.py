import os
import json
import torch
import copy
import shutil
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge models using Fisher Matrix and Sensitivity analysis.")
    parser.add_argument("--deepseek_path", type=str, required=True, help="Path to DeepSeek model")
    parser.add_argument("--meditron_path", type=str, required=True, help="Path to Meditron model")
    parser.add_argument("--qwen_path", type=str, required=True, help="Path to Qwen model")
    parser.add_argument("--fisher_matrix_path", type=str, required=True, help="Path to Fisher matrix .pt file")
    parser.add_argument("--json_samples_path", type=str, required=True, help="Path to JSON samples file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output model")
    parser.add_argument("--lambda_val", type=float, required=True, help="Lambda hyperparameter value")
    return parser.parse_args()

args = parse_arguments()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Use bfloat16 with device_map="auto" to save memory; switch to float32 if hardware doesn't support it
MODEL_DTYPE = torch.bfloat16

print(f"Using model dtype: {MODEL_DTYPE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")

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

# --- Helper Functions ---
def flatten_model_params(model, target_device='cpu'):
    """Flatten model parameters into a single vector and move to target CPU device."""
    # Parameters in a model loaded with device_map="auto" can be on different devices.
    # Move each to target_device (CPU) before concatenation.
    flat_params_on_target_device = [p.data.to(target_device, non_blocking=True).view(-1) for p in model.parameters()]
    return torch.cat(flat_params_on_target_device)

def unflatten_params_to_state_dict(flat_params_cpu, reference_model_auto_device):
    """
    Restore flattened CPU parameter vector to the model's state_dict structure.
    flat_params_cpu: Flattened parameters on CPU.
    reference_model_auto_device: Reference model loaded with device_map="auto" (params on various devices).
    """
    new_state_dict = OrderedDict()
    current_pos = 0
    for name, param in reference_model_auto_device.named_parameters():
        num_elements = param.numel()
        # param.device is the device of this specific parameter (due to device_map="auto")
        # Slice from CPU tensor, move to target device, then reshape
        try:
            chunk_on_cpu = flat_params_cpu[current_pos : current_pos + num_elements]
            # Move chunk to the original parameter's device before reshaping
            new_state_dict[name] = chunk_on_cpu.to(param.device, non_blocking=True).view_as(param.data).clone()
        except Exception as e:
            print(f"Error processing parameter {name} with shape {param.shape} and numel {num_elements}.")
            print(f"  param.device: {param.device}, flat_params_cpu device: {flat_params_cpu.device}")
            print(f"  Current position: {current_pos}, num_elements: {num_elements}, total flat_params: {len(flat_params_cpu)}")
            raise e
        current_pos += num_elements
    if current_pos != len(flat_params_cpu):
        raise ValueError(f"Size mismatch when unflattening parameters. Expected {len(flat_params_cpu)}, processed {current_pos}.")
    return new_state_dict

def load_model_and_flatten_params(model_path, params_target_device='cpu'):
    """Load model (device_map="auto") and return the model object and flattened CPU parameters."""
    print(f"Loading model from {model_path} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=MODEL_DTYPE, # Using bfloat16 for memory efficiency
        device_map="auto",
        trust_remote_code=True
    )
    model.eval() # Set to evaluation mode
    print(f"Model {model_path} loaded. Main device: {model.device}")
    with torch.no_grad():
        # Parameters will be moved to params_target_device (CPU) during flattening
        flat_params = flatten_model_params(model, target_device=params_target_device)
    print(f"Parameters for {model_path} flattened to {params_target_device}. Parameter count: {len(flat_params)}")
    return model, flat_params

# --- 1. Load models and flatten parameters to CPU ---
print("--- Step 1: Loading models and flattening parameters to CPU ---")
# Use try-except-finally to ensure memory is released even if errors occur
model_deepseek = None
meditron_model_for_grads = None # Will be loaded specifically for grads
temp_qwen_model = None

try:
    # model_deepseek is kept in memory as its structure is needed for unflattening later
    model_deepseek, model_deepseek_flat_cpu = load_model_and_flatten_params(args.deepseek_path, params_target_device='cpu')
    model_deepseek_flat_original_cpu = model_deepseek_flat_cpu.clone() # This is already on CPU

    # For Meditron and Qwen, we only need their flattened CPU parameters for calculations initially.
    # The Meditron model object for gradients will be loaded separately or re-used if memory allows.
    # To save VRAM, we load, flatten to CPU, then delete the model object if it's not the one for gradients.
    
    # Load Meditron, flatten its params to CPU
    # For now, let's assume meditron_model_for_grads will be this one.
    meditron_model_for_grads, model_meditron_flat_cpu = load_model_and_flatten_params(args.meditron_path, params_target_device='cpu')

    temp_qwen_model, model_qwen_flat_cpu = load_model_and_flatten_params(args.qwen_path, params_target_device='cpu')
    N = len(model_deepseek_flat_cpu)
    del temp_qwen_model # Qwen model object no longer needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"All models' parameters flattened to CPU. Parameter vector dimension N = {N}")

    if not (len(model_meditron_flat_cpu) == N and len(model_qwen_flat_cpu) == N):
        raise ValueError("Model parameter dimensions do not match! Ensure all models have the same architecture.")

    print(f"Loading Fisher matrix from {args.fisher_matrix_path} to CPU...")
    # fisher_matrix_flat_cpu = torch.load(args.fisher_matrix_path, map_location='cpu').view(-1) # Original line
    loaded_fisher_data = torch.load(args.fisher_matrix_path, map_location='cpu')

    if isinstance(loaded_fisher_data, list):
        if len(loaded_fisher_data) == 1 and isinstance(loaded_fisher_data[0], torch.Tensor):
            print("Fisher matrix was loaded as a list containing a single tensor. Using the first element.")
            fisher_matrix_flat_cpu = loaded_fisher_data[0].view(-1)
        # If your Fisher matrix was saved as a list of tensors that need to be concatenated:
        # elif all(isinstance(t, torch.Tensor) for t in loaded_fisher_data):
        #     print("Fisher matrix was loaded as a list of tensors. Concatenating them.")
        #     fisher_matrix_flat_cpu = torch.cat([t.view(-1) for t in loaded_fisher_data])
        else:
            raise TypeError(f"Fisher matrix file {args.fisher_matrix_path} loaded as a list, "
                            f"but it's not a list containing a single tensor as its first element, "
                            f"or a list of tensors. Please check the file contents. "
                            f"Number of elements: {len(loaded_fisher_data)}, "
                            f"Type of first element: {type(loaded_fisher_data[0]) if loaded_fisher_data else 'N/A'}")
    elif isinstance(loaded_fisher_data, torch.Tensor):
        print("Fisher matrix loaded as a single tensor.")
        fisher_matrix_flat_cpu = loaded_fisher_data.view(-1)
    else:
        raise TypeError(f"Fisher matrix file {args.fisher_matrix_path} did not load as a tensor or a recognized list format. "
                        f"Loaded type: {type(loaded_fisher_data)}")

    if len(fisher_matrix_flat_cpu) != N:
        raise ValueError(f"Fisher matrix dimension ({len(fisher_matrix_flat_cpu)}) after loading and processing "
                         f"does not match N ({N}).")
    print("Fisher matrix loaded to CPU and reshaped successfully.")

    # --- 2. Calculate Sensitivity Matrices (One N-dim CPU tensor per sample) ---
    print("\n--- Step 2: Calculating Sensitivity Matrices ---")
    with open(args.json_samples_path, "r", encoding="utf-8") as f:
        reason_samples = json.load(f)
    print(f"Loaded {len(reason_samples)} samples from {args.json_samples_path}.")

    raw_sensitivity_matrices_cpu = []

    # meditron_model_for_grads is already loaded with device_map="auto"
    tokenizer = AutoTokenizer.from_pretrained(args.meditron_path, trust_remote_code=True)
    
    meditron_model_for_grads.train() # Set to train mode for gradients
    # Create a state dict of original parameters. Tensors will be on their respective devices.
    original_meditron_state_dict_for_grads = {k: v.clone().detach() for k, v in meditron_model_for_grads.state_dict().items()}
    
    total_params_check_meditron = sum(p.numel() for p in meditron_model_for_grads.parameters())
    if total_params_check_meditron != N:
          print(f"Warning: Param count for Meditron gradient model ({total_params_check_meditron}) not equal to N ({N}).")


    for sample_idx, sample in enumerate(reason_samples):
        print(f"Processing sample {sample_idx+1}/{len(reason_samples)} for sensitivity...")
        sent1 = sample.get("question", "")
        prompt = prompt_template.format(sent1=sent1)
        # Tokenizer by default produces CPU tensors
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move inputs to the device of the first parameter of the model
        # model.device for device_map="auto" usually points to the device of the embedding layer
        input_ids = inputs["input_ids"].to(meditron_model_for_grads.device)
        labels = input_ids.clone()
        
        meditron_model_for_grads.zero_grad()
        
        outputs = meditron_model_for_grads(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"  Sample {sample_idx+1}: Loss = {loss.item():.4f} (Loss device: {loss.device})")
        
        loss.backward() # Gradients computed on devices holding model parts
        print(f"  Sample {sample_idx+1}: Gradients computed.")
        
        current_sample_sensitivity_flat_cpu = torch.zeros(N, dtype=torch.float32, device='cpu')
        
        current_pos = 0
        with torch.no_grad():
            for name, param in meditron_model_for_grads.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # original_value is on param.device (from state_dict copy)
                    original_value = original_meditron_state_dict_for_grads[name]
                    # Gradient (param.grad) is on param.device
                    # Calculation happens on param.device
                    sensitivity_val_for_param_on_device = param.grad.detach() * original_value.detach()
                    
                    numel_current_param = param.numel()
                    # Move result to CPU before placing into the CPU aggregate tensor
                    current_sample_sensitivity_flat_cpu[current_pos : current_pos + numel_current_param] = \
                        sensitivity_val_for_param_on_device.view(-1).cpu()
                    current_pos += numel_current_param
                elif param.requires_grad:
                    print(f"  Warning: Grad for trainable param {name} is None for sample {sample_idx+1}.")
                    current_pos += param.numel()
                else: # Not trainable or no grad but need to advance pointer
                    current_pos += param.numel()

        if current_pos != N and current_pos != total_params_check_meditron : # Check against N or actual model params
             print(f"  Warning: Mismatch in parameter accumulation for sample {sample_idx+1}. Expected {N} or {total_params_check_meditron}, got {current_pos}")

        raw_sensitivity_matrices_cpu.append(current_sample_sensitivity_flat_cpu)
        print(f"  Sample {sample_idx+1}: Sensitivity map generated and added (CPU).")
        del input_ids, labels, outputs, loss # Clean up GPU tensors for this sample
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


    del meditron_model_for_grads, original_meditron_state_dict_for_grads # Done with this model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Finished generating {len(raw_sensitivity_matrices_cpu)} raw sensitivity matrices on CPU.")

    # --- 3. Modify Sensitivity Matrices and Calculate Importance Tensor (Multi-GPU) ---
    print("\n--- Step 3: Modifying Sensitivities and Calculating Importance (Multi-GPU) ---")

    # param_diff_sq_cpu is already on CPU, and its original model was bfloat16, so it might be bfloat16
    # fisher_matrix_flat_cpu is on CPU, dtype depends on loading, assuming float32 or convertable
    # raw_sensitivity_matrices_cpu contains float32 tensors

    # Check CUDA device count
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA devices.")
    else:
        print("CUDA is not available. The requested multi-GPU strategy cannot be applied.")
        # Here we decide whether to raise an error or fallback.
        # For simplicity, if CUDA is missing, raise error as subsequent logic depends on it.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the multi-GPU processing step.")

    # Allocate devices based on availability
    # Plan: fisher * param_diff on cuda:0
    # result * LAMBDA on cuda:1
    # result + sensitivity on cuda:2
    
    if num_gpus < 1: # At least one GPU required
        raise RuntimeError("At least one GPU is required for the multi-GPU path.")

    device0 = 'cuda:0'
    device1 = 'cuda:1' if num_gpus >= 2 else device0
    device2 = 'cuda:2' if num_gpus >= 3 else device1 # If only 2 GPUs, device2 is cuda:1; if 1 GPU, device2 is cuda:0

    print(f"Using device mapping: product on {device0}, scaling on {device1}, addition on {device2}.")

    # param_diff_sq_cpu and fisher_matrix_flat_cpu are outside the loop, but moved to GPU inside,
    # because they are calculated against every sensitivity_matrix_cpu.
    
    # Ensure param_diff_sq_cpu started on CPU
    param_diff_sq_cpu = (model_deepseek_flat_cpu - model_meditron_flat_cpu)**2
    param_diff_sq_cpu = param_diff_sq_cpu.to('cpu', non_blocking=True)

    # Ensure fisher_matrix_flat_cpu is on CPU
    fisher_matrix_flat_cpu = fisher_matrix_flat_cpu.to('cpu', non_blocking=True)

    importance_tensor_cpu = torch.zeros(N, dtype=torch.int, device='cpu')

    for i, sensitivity_matrix_cpu in enumerate(raw_sensitivity_matrices_cpu):
        print(f"  Processing sensitivity matrix {i+1}/{len(raw_sensitivity_matrices_cpu)} using GPUs...")
        
        # sensitivity_matrix_cpu is currently on CPU, float32
        
        # 1. Move fisher_matrix_flat_cpu and param_diff_sq_cpu to device0 and calculate product
        #    param_diff_sq_cpu (bfloat16) * fisher_matrix_flat_cpu (float32) -> float32
        print(f"    Moving fisher and param_diff_sq to {device0} for product...")
        fisher_gpu0 = fisher_matrix_flat_cpu.to(device0, non_blocking=True)
        param_diff_sq_gpu0 = param_diff_sq_cpu.to(device0, non_blocking=True) 
        
        # Calculate product (on device0)
        # torch handles type promotion: bfloat16 * float32 -> float32
        product_gpu0 = fisher_gpu0 * param_diff_sq_gpu0
        print(f"    Product calculated on {device0}. Dtype: {product_gpu0.dtype}")

        # Release source tensors on device0 (if needed immediately)
        del fisher_gpu0
        del param_diff_sq_gpu0
        if device0 != device1: # Synchronize if devices differ
            torch.cuda.synchronize(device0)


        # 2. Move product to device1, then multiply by LAMBDA_VAL
        print(f"    Moving product from {device0} to {device1} for scaling...")
        product_gpu1 = product_gpu0.to(device1, non_blocking=True)
        del product_gpu0 # Release product_gpu0 on device0
        if device0 != device1 : torch.cuda.synchronize(device0)
        if device1 != device0 : torch.cuda.synchronize(device1)

        scaled_product_gpu1 = args.lambda_val * product_gpu1
        print(f"    Product scaled on {device1}. Dtype: {scaled_product_gpu1.dtype}")
        del product_gpu1 # Release product_gpu1 on device1
        if device1 != device2:
            torch.cuda.synchronize(device1)

        # 3. Move sensitivity_matrix_cpu to device2,
        #    Move scaled_product_gpu1 to device2 (if diff devices)
        #    Perform addition on device2
        print(f"    Moving sensitivity_matrix and scaled_product to {device2} for addition...")
        sensitivity_gpu2 = sensitivity_matrix_cpu.to(device2, non_blocking=True) # float32
        
        scaled_product_gpu2 = scaled_product_gpu1.to(device2, non_blocking=True)
        del scaled_product_gpu1 # Release scaled_product_gpu1 on device1
        if device1 != device2 : torch.cuda.synchronize(device1)
        if device2 != device1 : torch.cuda.synchronize(device2)
        
        modified_sensitivity_gpu2 = sensitivity_gpu2 + scaled_product_gpu2
        print(f"    Addition performed on {device2}. Dtype: {modified_sensitivity_gpu2.dtype}")
        
        # Release components on device2
        del sensitivity_gpu2
        del scaled_product_gpu2
        torch.cuda.synchronize(device2) # Ensure computation complete

        # 4. Move final result back to CPU
        print(f"    Moving result from {device2} to CPU...")
        modified_sensitivity_on_cpu = modified_sensitivity_gpu2.cpu() # Blocking operation
        del modified_sensitivity_gpu2 # Release final GPU memory
        
        # 5. Calculate binarized_sensitivity and update importance_tensor_cpu on CPU
        binarized_sensitivity_cpu = torch.where(modified_sensitivity_on_cpu <= 0, 1, -1)
        importance_tensor_cpu += binarized_sensitivity_cpu
        
        # 6. Delete tensors no longer needed in this loop iteration
        del modified_sensitivity_on_cpu
        del binarized_sensitivity_cpu
        # 'sensitivity_matrix_cpu' is a reference from raw_sensitivity_matrices_cpu
        # deleting it here only removes the local reference.

        print(f"  Finished processing and updating importance for sensitivity matrix {i+1}.")
        # Clean GPU cache at the end of iteration
        if torch.cuda.is_available():
            for gpu_idx in range(num_gpus):
                with torch.cuda.device(f'cuda:{gpu_idx}'):
                    torch.cuda.empty_cache()


    print("Importance tensor calculated on CPU using multi-GPU for intermediate steps.")
    # Delete large CPU tensors created before loop
    del param_diff_sq_cpu 
    del raw_sensitivity_matrices_cpu # Assuming raw list is no longer needed

    # --- Subsequent steps (4, 5) continue on CPU ---

    # --- 4. Merging Model Weights (CPU) ---
    print("\n--- Step 4: Merging Model Weights (CPU) ---")
    
    model_diff_med_qwen_flat_cpu = model_meditron_flat_cpu - model_qwen_flat_cpu
    merged_model_flat_cpu = model_deepseek_flat_original_cpu.clone() # Start with DeepSeek original (CPU)
    
    update_indices_cpu = torch.where(importance_tensor_cpu > 0)[0]
    print(f"  Number of parameters to update based on importance: {len(update_indices_cpu)} out of {N}")
    
    merged_model_flat_cpu[update_indices_cpu] += model_diff_med_qwen_flat_cpu[update_indices_cpu]
    print("Model weights merged on CPU.")
    del model_diff_med_qwen_flat_cpu, importance_tensor_cpu, update_indices_cpu
    # model_deepseek_flat_original_cpu, model_meditron_flat_cpu, model_qwen_flat_cpu can be cleared if no longer needed
    # merged_model_flat_cpu is the final result on CPU

    # --- 5. Saving Merged Model and Configuration ---
    print("\n--- Step 5: Saving Merged Model and Configuration ---")
    
    # model_deepseek (loaded with device_map="auto") is used as reference for structure and devices
    # merged_model_flat_cpu is on CPU
    print("Unflattening merged parameters back to state_dict with original device distribution...")
    merged_state_dict = unflatten_params_to_state_dict(merged_model_flat_cpu, model_deepseek)
    
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin") # HF usually expects this name
    torch.save(merged_state_dict, output_model_file)
    print(f"Merged model state_dict saved to {output_model_file}")

    # Copy config files from the DeepSeek model directory (or any base model)
    config_source_dir = args.deepseek_path
    files_to_copy = ["config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    # Add any other files like generation_config.json if they exist and are needed
    optional_files = ["generation_config.json", "vocab.json", "merges.txt"] # For some tokenizers

    for filename in files_to_copy + optional_files:
        src_file = os.path.join(config_source_dir, filename)
        dst_file = os.path.join(args.output_dir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {filename} to {args.output_dir}")
        elif filename in files_to_copy: # Only warn if a mandatory file is missing
             print(f"Warning: Source file {src_file} not found, not copied.")
            
    print(f"\nProcess completed. Merged model and config files are in: {args.output_dir}")
    print(f"You should now be able to load the merged model using: AutoModelForCausalLM.from_pretrained('{args.output_dir}', device_map='auto', trust_remote_code=True)")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Cleaning up model objects...")
    del model_deepseek
    # meditron_model_for_grads already deleted
    # temp_qwen_model already deleted
    # Delete CPU tensors if they are large and no longer needed by this point
    if 'model_deepseek_flat_cpu' in locals(): del model_deepseek_flat_cpu
    if 'model_deepseek_flat_original_cpu' in locals(): del model_deepseek_flat_original_cpu
    if 'model_meditron_flat_cpu' in locals(): del model_meditron_flat_cpu
    if 'model_qwen_flat_cpu' in locals(): del model_qwen_flat_cpu
    if 'fisher_matrix_flat_cpu' in locals(): del fisher_matrix_flat_cpu
    if 'merged_model_flat_cpu' in locals(): del merged_model_flat_cpu
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleanup complete.")