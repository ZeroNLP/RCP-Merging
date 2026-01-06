import os
import json
import torch
import copy
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

# --- 0. Configure relevant paths and model names ---
# Main model path
PATH_DEEPSEEK = "/data/transformers/DeepSeek-R1-Distill-Qwen-7B/"
PATH_MEDITRON = "/data/transformers/Meditron3-Qwen2.5-7B/"
PATH_QWEN = "/data/transformers/Qwen2.5-7B/"

# Fisher matrix path
FISHER_MATRIX_PATH = "/data/jyyang/model_merge/EWC_Fisher/Fisher_matrix/Fisher_matrix.pt"

# Output path
OUTPUT_DIR = "/data/jyyang/model_merge/mergekit/merged_models/two_models/Medical/EWC/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_file = os.path.join(OUTPUT_DIR, "sensitivity_matrix.pt")
# JSON sample file location
JSON_SAMPLES_PATH = "path to medical_samples.json"

# Hyperparameters
LAMBDA_VAL = 1000.0

# Use bfloat16 with device_map="auto" to save memory, can be switched back to float32 if hardware does not support it
MODEL_DTYPE = torch.bfloat16

print(f"Using model dtype: {MODEL_DTYPE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() == 0:
        print("Warning: torch.cuda.is_available() is True, but device count is 0. GPU operations will fail.")
else:
    print("Warning: CUDA is not available. The second part of the script which requires CUDA will fail.")


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
    """Flattens model parameters into a single vector and places it on the target CPU device"""
    flat_params_on_target_device = []
    for p in model.parameters():
        flat_params_on_target_device.append(p.data.to(target_device, non_blocking=True).view(-1))
    return torch.cat(flat_params_on_target_device)

def unflatten_params_to_state_dict(flat_params_on_source_device, reference_model_auto_device):
    """
    Restores the flattened parameter vector to the model's state_dict.
    flat_params_on_source_device: The flattened parameters on their current device.
    reference_model_auto_device: The reference model loaded with device_map="auto", with its parameters on different devices.
    """
    new_state_dict = OrderedDict()
    current_pos = 0
    # Ensure flat_params is on CPU for slicing if it's large and needs to be sent piece by piece.
    # However, if it's already on a GPU and fits, slicing there is fine.
    # For this function, let's assume flat_params_on_source_device can be on CPU or GPU.
    # The to(param.device) handles moving the chunk.
    
    for name, param in reference_model_auto_device.named_parameters():
        num_elements = param.numel()
        try:
            # Slice from the flat_params (which could be on CPU or GPU)
            chunk_on_source_device = flat_params_on_source_device[current_pos : current_pos + num_elements]
            # Move chunk to the original parameter's device before reshaping
            new_state_dict[name] = chunk_on_source_device.to(param.device, non_blocking=True).view_as(param.data).clone()
        except Exception as e:
            print(f"Error processing parameter {name} with shape {param.shape} and numel {num_elements}.")
            print(f"  param.device: {param.device}, flat_params device: {flat_params_on_source_device.device}")
            print(f"  Current position: {current_pos}, num_elements: {num_elements}, total flat_params: {len(flat_params_on_source_device)}")
            raise e
        current_pos += num_elements
    if current_pos != len(flat_params_on_source_device):
        raise ValueError(f"Size mismatch when unflattening parameters. Expected {len(flat_params_on_source_device)}, processed {current_pos}.")
    return new_state_dict

def load_model_and_flatten_params(model_path, params_target_device='cpu'):
    """Loads a model (device_map="auto") and returns the model object and its parameters flattened to CPU"""
    print(f"Loading model from {model_path} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=MODEL_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"Model {model_path} loaded. Main device: {model.device}") # model.device shows the device of the first module
    # For parameters:
    # devices = {p.device for p in model.parameters()}
    # print(f"  Parameters are on devices: {devices}")

    with torch.no_grad():
        flat_params = flatten_model_params(model, target_device=params_target_device)
    print(f"Parameters for {model_path} flattened to {params_target_device}. Parameter count: {len(flat_params)}")
    return model, flat_params

# --- Global variables to hold data between parts ---
# These would be populated by the end of Part 1
model_deepseek_obj = None
model_deepseek_flat_params_cpu = None
model_deepseek_flat_original_cpu = None # This is a clone of model_deepseek_flat_params_cpu
model_meditron_flat_params_cpu = None
model_qwen_flat_params_cpu = None
fisher_matrix_flat_cpu = None
raw_sensitivity_matrices_cpu = []
N = 0

# --- 1. Load models and flatten parameters to CPU ---
print("--- Step 1: Loading models and flattening parameters to CPU ---")
# Use a broader try-except-finally for cleanup if Part 1 is run standalone
temp_meditron_model_for_grads_obj = None # Renamed to avoid confusion with the one in step 2
temp_qwen_model_obj = None

try:
    model_deepseek_obj, model_deepseek_flat_params_cpu = load_model_and_flatten_params(PATH_DEEPSEEK, params_target_device='cpu')
    model_deepseek_flat_original_cpu = model_deepseek_flat_params_cpu.clone() # Keep an original copy on CPU

    # Load Meditron, flatten its params to CPU
    # We'll load a fresh Meditron model for gradient calculation in Step 2 to manage memory better
    # So, the model object from this load can be deleted after flattening.
    temp_meditron_obj, model_meditron_flat_params_cpu = load_model_and_flatten_params(PATH_MEDITRON, params_target_device='cpu')
    del temp_meditron_obj # Delete model object to save VRAM
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    temp_qwen_model_obj, model_qwen_flat_params_cpu = load_model_and_flatten_params(PATH_QWEN, params_target_device='cpu')
    N = len(model_deepseek_flat_params_cpu)
    del temp_qwen_model_obj # Qwen model object no longer needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"All models' parameters flattened to CPU. Parameter vector dimension N = {N}")

    if not (len(model_meditron_flat_params_cpu) == N and len(model_qwen_flat_params_cpu) == N):
        raise ValueError("Model parameter dimensions do not match! Ensure all models have the same architecture.")

    print(f"Loading Fisher matrix from {FISHER_MATRIX_PATH} to CPU...")
    loaded_fisher_data = torch.load(FISHER_MATRIX_PATH, map_location='cpu')

    if isinstance(loaded_fisher_data, list):
        if len(loaded_fisher_data) == 1 and isinstance(loaded_fisher_data[0], torch.Tensor):
            print("Fisher matrix was loaded as a list containing a single tensor. Using the first element.")
            fisher_matrix_flat_cpu = loaded_fisher_data[0].view(-1)
        else:
            raise TypeError(f"Fisher matrix file {FISHER_MATRIX_PATH} loaded as a list, "
                            f"but it's not a list containing a single tensor as its first element. "
                            f"Number of elements: {len(loaded_fisher_data)}, "
                            f"Type of first element: {type(loaded_fisher_data[0]) if loaded_fisher_data else 'N/A'}")
    elif isinstance(loaded_fisher_data, torch.Tensor):
        print("Fisher matrix loaded as a single tensor.")
        fisher_matrix_flat_cpu = loaded_fisher_data.view(-1)
    else:
        raise TypeError(f"Fisher matrix file {FISHER_MATRIX_PATH} did not load as a tensor or a recognized list format. "
                        f"Loaded type: {type(loaded_fisher_data)}")

    if len(fisher_matrix_flat_cpu) != N:
        raise ValueError(f"Fisher matrix dimension ({len(fisher_matrix_flat_cpu)}) "
                         f"does not match N ({N}).")
    print("Fisher matrix loaded to CPU and reshaped successfully.")

    # --- 2. Calculate Sensitivity Matrices (one N-dimensional CPU tensor per sample) ---
    print("\n--- Step 2: Calculating Sensitivity Matrices ---")
    with open(JSON_SAMPLES_PATH, "r", encoding="utf-8") as f:
        reason_samples = json.load(f)
    print(f"Loaded {len(reason_samples)} samples from {JSON_SAMPLES_PATH}.")

    # Load Meditron model specifically for gradient calculation
    # This ensures that device_map="auto" places it optimally if VRAM was freed up
    print(f"Loading Meditron model ({PATH_MEDITRON}) again for gradient calculations...")
    meditron_model_for_grads_obj = AutoModelForCausalLM.from_pretrained(
        PATH_MEDITRON,
        torch_dtype=MODEL_DTYPE,
        device_map="auto", # device_map="auto" is crucial for multi-GPU or large models
        trust_remote_code=True
    )
    meditron_model_for_grads_obj.eval() # Set to eval mode initially
    print(f"Meditron model for gradients loaded. Main device: {meditron_model_for_grads_obj.device}")

    tokenizer = AutoTokenizer.from_pretrained(PATH_MEDITRON, trust_remote_code=True)
    
    meditron_model_for_grads_obj.train() # Set to train mode for gradients
    # Create a state dict of original parameters. Tensors will be on their respective devices.
    original_meditron_state_dict_for_grads = {
        k: v.clone().detach() for k, v in meditron_model_for_grads_obj.state_dict().items()
    }
    
    total_params_check_meditron = sum(p.numel() for p in meditron_model_for_grads_obj.parameters())
    if total_params_check_meditron != N:
         print(f"Warning: Param count for Meditron gradient model ({total_params_check_meditron}) not equal to N ({N}). This might indicate an issue if N was derived from a different model structure or if this model is different.")


    for sample_idx, sample in enumerate(reason_samples):
        print(f"Processing sample {sample_idx+1}/{len(reason_samples)} for sensitivity...")
        sent1 = sample.get("question", "")
        prompt = prompt_template.format(sent1=sent1)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        input_ids = inputs["input_ids"].to(meditron_model_for_grads_obj.device) # Move to model's input device
        labels = input_ids.clone()
        
        meditron_model_for_grads_obj.zero_grad()
        
        outputs = meditron_model_for_grads_obj(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"  Sample {sample_idx+1}: Loss = {loss.item():.4f} (Loss device: {loss.device})")
        
        loss.backward()
        print(f"  Sample {sample_idx+1}: Gradients computed.")
        
        current_sample_sensitivity_flat_cpu_list = []
        with torch.no_grad():
            for name, param in meditron_model_for_grads_obj.named_parameters():
                if param.requires_grad and param.grad is not None:
                    original_value_on_param_device = original_meditron_state_dict_for_grads[name]
                    sensitivity_val_for_param_on_device = param.grad.detach() * original_value_on_param_device.detach()
                    current_sample_sensitivity_flat_cpu_list.append(sensitivity_val_for_param_on_device.view(-1).cpu())
                elif param.requires_grad and param.grad is None:
                    print(f"  Warning: Grad for trainable param {name} is None for sample {sample_idx+1}. Appending zeros.")
                    current_sample_sensitivity_flat_cpu_list.append(torch.zeros(param.numel(), dtype=torch.float32, device='cpu'))
                else: # Not trainable, but need to maintain order if concatenating all params
                    # This part is tricky if N is based on all params but some are not trainable.
                    # The original flatten_model_params takes all model.parameters().
                    # Assuming N corresponds to all parameters, and we need a sensitivity value (even 0) for all.
                    # However, the original code only advances current_pos. Let's stick to a list and then cat.
                    # If a param is not part of "N" because it's not trainable, it shouldn't be here.
                    # The original code implies N is sum of numel of all params.
                     if param.numel() > 0 : # Only add if it has elements
                        # if not param.requires_grad, it means it won't have a gradient.
                        # The original code just advanced current_pos.
                        # If we are building a flat tensor of size N, we need to fill these spots.
                        # Typically, sensitivity for non-trained params is 0.
                        # Let's ensure the sum of numels matches N.
                        # The original code's current_pos logic implicitly handles this by matching N.
                        # Here, by appending to list and then cat, we must be careful.
                        # Best to only append sensitivity for params that *contribute* to the flat N.
                        # The flatten_model_params function includes all parameters.
                        # So, if a parameter doesn't have a gradient (e.g. not requires_grad, or error),
                        # we should still account for its size with zeros.
                        if not param.requires_grad: # Fill with zeros for non-trainable params
                             current_sample_sensitivity_flat_cpu_list.append(torch.zeros(param.numel(), dtype=torch.float32, device='cpu'))


        current_sample_sensitivity_flat_cpu = torch.cat(current_sample_sensitivity_flat_cpu_list)
        if len(current_sample_sensitivity_flat_cpu) != N:
            print(f"  Error: Length of concatenated sensitivity ({len(current_sample_sensitivity_flat_cpu)}) for sample {sample_idx+1} does not match N ({N}). Check parameter handling.")
            # Fallback: create a zero tensor of size N to avoid breaking the list structure
            # This indicates a problem in sensitivity calculation or parameter matching.
            current_sample_sensitivity_flat_cpu = torch.zeros(N, dtype=torch.float32, device='cpu')


        raw_sensitivity_matrices_cpu.append(current_sample_sensitivity_flat_cpu)
        print(f"  Sample {sample_idx+1}: Sensitivity map generated and added (CPU). Length: {len(current_sample_sensitivity_flat_cpu)}")
        del input_ids, labels, outputs, loss, current_sample_sensitivity_flat_cpu_list, current_sample_sensitivity_flat_cpu
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.save(raw_sensitivity_matrices_cpu, save_file)

    del meditron_model_for_grads_obj, original_meditron_state_dict_for_grads # Done with this model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Finished generating {len(raw_sensitivity_matrices_cpu)} raw sensitivity matrices on CPU.")
    print("Part 1 Complete. Necessary tensors are on CPU and 'model_deepseek_obj' is loaded.")

except Exception as e:
    print(f"An error occurred in Part 1: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up temporary models used only in Part 1 if they haven't been deleted
    if 'temp_meditron_obj' in locals() and temp_meditron_obj is not None: del temp_meditron_obj
    if 'temp_qwen_model_obj' in locals() and temp_qwen_model_obj is not None: del temp_qwen_model_obj
    if 'meditron_model_for_grads_obj' in locals() and meditron_model_for_grads_obj is not None: del meditron_model_for_grads_obj
        
    # Note: 'model_deepseek_obj' and the flat param tensors are intentionally kept for Part 2.
    # Their cleanup will be handled by Part 2's finally block or at the end of the entire script.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Part 1 cleanup attempted for temporary objects.")