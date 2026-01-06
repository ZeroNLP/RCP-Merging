import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
import copy

# --- Configuration Loader ---
def get_env_var(key, default=None, cast_func=str):
    """Helper to fetch and cast environment variables safely."""
    val = os.getenv(key)
    if val is None:
        if default is not None:
            return default
        else:
            raise ValueError(f"Environment variable '{key}' is required but not set.")
    return cast_func(val)

# --- Hyperparameters Setup ---
try:
    BASE_MODEL_PATH = get_env_var("BASE_MODEL_PATH")
    MODEL_1_PATH = get_env_var("MODEL_1_PATH")
    MODEL_1_NAME = get_env_var("MODEL_1_NAME")
    MODEL_2_PATH = get_env_var("MODEL_2_PATH")
    MODEL_2_NAME = get_env_var("MODEL_2_NAME")
    MERGED_MODEL_SAVE_PATH = get_env_var("MERGED_MODEL_SAVE_PATH")

    REASONING_SAMPLES_PATH = get_env_var("REASONING_SAMPLES_PATH")
    MEDICAL_SAMPLES_PATH = get_env_var("MEDICAL_SAMPLES_PATH")

    # Casting numeric hyperparameters
    LAMBDA_CONFLICT_PRESERVE = get_env_var("LAMBDA_CONFLICT_PRESERVE", cast_func=float)
    C_DIM_TRIM = get_env_var("C_DIM_TRIM", cast_func=int)
    ALPHA_TASK_VECTOR_SCALE = get_env_var("ALPHA_TASK_VECTOR_SCALE", cast_func=float)
    NUM_EXEMPLARS_PER_TASK = get_env_var("NUM_EXEMPLARS_PER_TASK", cast_func=int)
    
    SYS_PROMPT_REASONING = get_env_var("SYS_PROMPT_REASONING")

except ValueError as e:
    print(f"Configuration Error: {e}")
    sys.exit(1)

# Fixed Technical Configurations
TORCH_DTYPE = torch.bfloat16
LINALG_DTYPE = torch.float32  # Float32 required for stability in eigh/svd/lobpcg

# --- Utility Functions ---
def load_model_and_tokenizer(model_path):
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        primary_device = next(model.parameters()).device
    except StopIteration: 
        primary_device = torch.device("cpu")
    except AttributeError:
        if torch.cuda.is_available():
            primary_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            primary_device = torch.device("cpu")
            
    print(f"Model {model_path} loaded. Primary parameter device: {primary_device}")
    return model, tokenizer

def get_task_vector(finetuned_model_state_dict, base_model_state_dict):
    """Calculates Task Vector: T = Theta_fine - Theta_base"""
    task_vector = OrderedDict()
    for key, finetuned_param in finetuned_model_state_dict.items():
        if key in base_model_state_dict:
            base_param = base_model_state_dict[key].to(device=finetuned_param.device, dtype=finetuned_param.dtype)
            task_vector[key] = finetuned_param.clone() - base_param
            del base_param  # Free memory
        else:
            print(f"Warning: Key {key} found in finetuned model but not in base model. Cloning finetuned param.")
            task_vector[key] = finetuned_param.clone()
    return task_vector

# --- Data Loading for Exemplars ---
def load_json_samples(file_path, num_samples):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:num_samples] if data else []
    except FileNotFoundError:
        print(f"Error: Exemplar file not found at {file_path}")
        return []

def get_reasoning_exemplars(tokenizer, num_samples, device):
    samples_data = load_json_samples(REASONING_SAMPLES_PATH, num_samples)
    if not samples_data: return None
    prompts = []
    for sample in samples_data:
        input_reannotated = sample.get("reannotated_assistant_content", "")
        input_problem = sample.get("problem", "")
        input_solution = sample.get("solution", "")
        # Construct prompt following the instruction tuning format
        input_prompt = (
            f"[INST] <<SYS>>\n{SYS_PROMPT_REASONING}\n<</SYS>>\n\n"
            f"Problem:\n{input_problem}\nModel Answer:\n{input_reannotated}\n"
            f"Solution:\n{input_solution}\n[/INST]"
        )
        prompts.append(input_prompt)
    if not prompts: return None
    tokenized_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return {k: v.to(device) for k, v in tokenized_inputs.items()}

def get_medical_exemplars(tokenizer, num_samples, device):
    samples_data = load_json_samples(MEDICAL_SAMPLES_PATH, num_samples)
    if not samples_data: return None
    prompts = [sample.get("question", "") for sample in samples_data if sample.get("question")]
    if not prompts: return None
    tokenized_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return {k: v.to(device) for k, v in tokenized_inputs.items()}

# --- CAT Merging Core Logic (Memory Optimized) ---
def compute_trimming_basis_or_mask(param_name, param_type, T_i_layer, T_k_layer, X_i_features_S, X_k_features_S, lambda_val, c_val):
    """
    Computes the projection basis (B_k) or mask (m_k) to trim conflicting dimensions.
    """
    if T_i_layer is None:
        return None

    current_device = T_i_layer.device
    T_i_layer_linalg = T_i_layer.to(dtype=LINALG_DTYPE)
    B_k_or_m_k = None

    if param_type == 'linear_weight':
        if X_k_features_S is None or X_i_features_S is None:
            # print(f"Skipping trim for {param_name}: Missing S features.")
            return None
        
        # Move feature matrices to the correct device
        X_k_S_linalg = X_k_features_S.to(device=current_device, dtype=LINALG_DTYPE)
        X_i_S_linalg = X_i_features_S.to(device=current_device, dtype=LINALG_DTYPE)

        # Validate shapes
        if T_i_layer_linalg.shape[1] != X_k_S_linalg.shape[0]:
            return None

        try:
            # Calculate difference in covariance matrices
            S_diff = X_k_S_linalg - lambda_val * X_i_S_linalg
            
            # Compute Gradient-like matrix G
            matrix_G = T_i_layer_linalg @ S_diff @ T_i_layer_linalg.T
            del S_diff
            
            if torch.isnan(matrix_G).any() or torch.isinf(matrix_G).any():
                del matrix_G
                return None
            
            # Ensure symmetry
            matrix_G = (matrix_G + matrix_G.T) / 2.0

            num_eigvecs_available = matrix_G.shape[0]
            actual_c = min(c_val, num_eigvecs_available // 2 if num_eigvecs_available > 1 else 1)
            actual_c = max(1, actual_c)

            computed_B_k = None
            try:
                # 1. Try LOBPCG (Fastest for top-k eigenvalues on large sparse/structured matrices)
                eigvals_lobpcg, eigvecs_lobpcg = torch.lobpcg(matrix_G, k=actual_c, largest=True, niter=-1)
                computed_B_k = eigvecs_lobpcg
            except Exception:
                try:
                    # 2. Fallback to standard eigh on GPU
                    eigvals, eigvecs = torch.linalg.eigh(matrix_G)
                    computed_B_k = eigvecs[:, -actual_c:]
                except Exception:
                    # 3. Fallback to CPU if GPU runs out of memory (OOM)
                    del matrix_G
                    torch.cuda.empty_cache()
                    try:
                        matrix_G_cpu = T_i_layer_linalg.cpu() @ \
                                       (X_k_S_linalg.cpu() - lambda_val * X_i_S_linalg.cpu()) @ \
                                       T_i_layer_linalg.T.cpu()
                        matrix_G_cpu = (matrix_G_cpu + matrix_G_cpu.T) / 2.0
                        
                        eigvals_cpu, eigvecs_cpu = torch.linalg.eigh(matrix_G_cpu)
                        computed_B_k = eigvecs_cpu[:, -actual_c:].to(current_device)
                        del matrix_G_cpu, eigvals_cpu, eigvecs_cpu
                    except Exception:
                        return None 
            
            B_k_or_m_k = computed_B_k
            if 'matrix_G' in locals(): del matrix_G

        except RuntimeError:
            return None

    elif param_type in ['norm_scale', 'norm_shift']:
        if T_i_layer_linalg.ndim != 1: return None

        if param_type == 'norm_scale':
            if X_k_features_S is None or X_i_features_S is None: return None
            # Element-wise conflict score
            g_z = (X_k_features_S.to(device=current_device, dtype=LINALG_DTYPE) * (T_i_layer_linalg**2)) - \
                  lambda_val * (X_i_features_S.to(device=current_device, dtype=LINALG_DTYPE) * (T_i_layer_linalg**2))
        else: # norm_shift
            g_z = T_i_layer_linalg**2
        
        actual_c_norm = min(c_val, len(g_z))
        if actual_c_norm <= 0: return None

        # Select top-k conflicting indices
        _, top_indices = torch.topk(g_z, k=actual_c_norm)
        m_k_res = torch.zeros_like(T_i_layer_linalg, dtype=torch.bool, device=current_device)
        m_k_res[top_indices] = True
        B_k_or_m_k = m_k_res
        del g_z, top_indices
    
    return B_k_or_m_k

def trim_task_vector_component(T_i_layer, basis_or_mask, param_type):
    """Applies the calculated projection basis or mask to the task vector."""
    if basis_or_mask is None or T_i_layer is None:
        return T_i_layer

    device = T_i_layer.device
    dtype = T_i_layer.dtype
    
    basis_or_mask_prep = basis_or_mask.to(device=device)
    if basis_or_mask.is_floating_point():
        basis_or_mask_prep = basis_or_mask_prep.to(dtype=dtype)

    if param_type == 'linear_weight':
        # Project out the conflicting subspace
        # Formula: T_new = T - (B B^T) T
        projection_matrix = basis_or_mask_prep @ basis_or_mask_prep.T 
        subtracted_term = projection_matrix @ T_i_layer
        result = T_i_layer - subtracted_term
        del projection_matrix, subtracted_term, basis_or_mask_prep
        return result
            
    elif param_type in ['norm_scale', 'norm_shift']:
        # Zero out the masked elements
        result = T_i_layer - (T_i_layer * basis_or_mask_prep.to(dtype)) 
        del basis_or_mask_prep
        return result
    
    return T_i_layer

def get_param_type(param_name, param_tensor):
    """Classifies parameters to determine which trimming strategy to apply."""
    if 'norm.weight' in param_name or ('layernorm.weight' in param_name and '.attention.output_layernorm.weight' not in param_name):
        return 'norm_scale'
    if '.input_layernorm.weight' in param_name or '.post_attention_layernorm.weight' in param_name:
         return 'norm_scale'
    if 'norm.bias' in param_name or 'layernorm.bias' in param_name: 
        return 'norm_shift'
    if param_tensor.ndim == 2 and 'weight' in param_name:
        if 'embed_tokens.weight' in param_name or 'lm_head.weight' in param_name: 
            return 'other' 
        return 'linear_weight'
    if param_tensor.ndim == 1 and 'bias' in param_name:
        return 'norm_shift' 
    return 'other'

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting CAT Merging Pipeline ---")
    if torch.cuda.is_available():
        print(f"CUDA Active. Devices: {torch.cuda.device_count()}")
    else:
        print("WARNING: Running on CPU.")

    # 1. Load Models
    base_model, tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH)
    model1_reasoning, _ = load_model_and_tokenizer(MODEL_1_PATH)
    model2_medical, _ = load_model_and_tokenizer(MODEL_2_PATH)
    
    base_sd = base_model.state_dict()
    model1_sd = model1_reasoning.state_dict()
    model2_sd = model2_medical.state_dict()

    # 2. Calculate Task Vectors (TV = Model_Fine - Model_Base)
    print("Computing Task Vectors...")
    task_vector1 = get_task_vector(model1_sd, base_sd)
    task_vector2 = get_task_vector(model2_sd, base_sd)
    
    # Cleanup heavy model objects
    del model1_reasoning, model2_medical, model1_sd, model2_sd
    print("Original models unloaded to free memory.")
    torch.cuda.empty_cache()

    # NOTE: Real feature extraction logic should be inserted here using the exemplars functions.
    # Currently using placeholder identity matrices for demonstration stability.

    print("--- Beginning Task Vector Trimming ---")
    edited_task_vector1 = copy.deepcopy(task_vector1)
    edited_task_vector2 = copy.deepcopy(task_vector2)
    del task_vector1, task_vector2
    torch.cuda.empty_cache()
    
    param_names = list(base_sd.keys())

    # --- Loop 1: Edit TV2 (Medical) relative to TV1 (Reasoning) ---
    print(f"\nPhase 1: Refining {MODEL_2_NAME} (TV2) against {MODEL_1_NAME} (TV1)...")
    for i, name in enumerate(param_names):
        print(f"  Processing {i+1}/{len(param_names)}: {name}", end='\r')
        if name not in edited_task_vector2 or name not in edited_task_vector1: continue
            
        param_tv2_i = edited_task_vector2[name] 
        param_tv1_k = edited_task_vector1[name] 
        target_device = param_tv2_i.device 
        param_type = get_param_type(name, param_tv2_i)

        # Placeholder: Using Identity Matrices for Covariance (Replace with real feature extraction)
        cur_S_k, cur_S_i = None, None
        if param_type == 'linear_weight':
            in_dim = param_tv2_i.shape[1] 
            cur_S_k = torch.eye(in_dim, device=target_device, dtype=LINALG_DTYPE) * 0.5 
            cur_S_i = torch.eye(in_dim, device=target_device, dtype=LINALG_DTYPE) * 0.4
        elif param_type == 'norm_scale':
            dim_norm = param_tv2_i.shape[0]
            cur_S_k = torch.ones(dim_norm, device=target_device, dtype=LINALG_DTYPE) * 0.5
            cur_S_i = torch.ones(dim_norm, device=target_device, dtype=LINALG_DTYPE) * 0.4

        if param_type != 'other':
            basis_or_mask = compute_trimming_basis_or_mask(
                name, param_type, param_tv2_i, param_tv1_k.to(target_device),
                cur_S_i, cur_S_k, LAMBDA_CONFLICT_PRESERVE, C_DIM_TRIM
            )
            if basis_or_mask is not None:
                edited_task_vector2[name] = trim_task_vector_component(param_tv2_i, basis_or_mask, param_type)
                del basis_or_mask
        
        del cur_S_k, cur_S_i
        if (i+1) % 10 == 0: torch.cuda.empty_cache()

    # --- Loop 2: Edit TV1 (Reasoning) relative to TV2 (Medical) ---
    print(f"\nPhase 2: Refining {MODEL_1_NAME} (TV1) against {MODEL_2_NAME} (TV2)...")
    for i, name in enumerate(param_names):
        print(f"  Processing {i+1}/{len(param_names)}: {name}", end='\r')
        if name not in edited_task_vector1 or name not in edited_task_vector2: continue

        param_tv1_i = edited_task_vector1[name]
        param_tv2_k = edited_task_vector2[name]
        target_device = param_tv1_i.device
        param_type = get_param_type(name, param_tv1_i)
        
        # Placeholder Covariance
        cur_S_k, cur_S_i = None, None
        if param_type == 'linear_weight':
            in_dim = param_tv1_i.shape[1]
            cur_S_k = torch.eye(in_dim, device=target_device, dtype=LINALG_DTYPE) * 0.4
            cur_S_i = torch.eye(in_dim, device=target_device, dtype=LINALG_DTYPE) * 0.5
        elif param_type == 'norm_scale':
            dim_norm = param_tv1_i.shape[0]
            cur_S_k = torch.ones(dim_norm, device=target_device, dtype=LINALG_DTYPE) * 0.4
            cur_S_i = torch.ones(dim_norm, device=target_device, dtype=LINALG_DTYPE) * 0.5

        if param_type != 'other':
            basis_or_mask = compute_trimming_basis_or_mask(
                name, param_type, param_tv1_i, param_tv2_k.to(target_device), 
                cur_S_i, cur_S_k, LAMBDA_CONFLICT_PRESERVE, C_DIM_TRIM
            )
            if basis_or_mask is not None:
                edited_task_vector1[name] = trim_task_vector_component(param_tv1_i, basis_or_mask, param_type)
                del basis_or_mask
        
        del cur_S_k, cur_S_i
        if (i+1) % 10 == 0: torch.cuda.empty_cache()

    # 3. Final Merge
    print("\n--- Merging Edited Task Vectors with Base Model ---")
    final_merged_sd = OrderedDict()
    all_keys = set(base_sd.keys()) | set(edited_task_vector1.keys()) | set(edited_task_vector2.keys())

    for key in all_keys:
        # Resolve device/dtype based on availability
        if key in base_model.state_dict():
            target_device = base_model.state_dict()[key].device
            target_dtype = base_model.state_dict()[key].dtype
        elif key in edited_task_vector1:
            target_device = edited_task_vector1[key].device
            target_dtype = edited_task_vector1[key].dtype
        else:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            target_dtype = TORCH_DTYPE
        
        # W_merged = W_base + alpha*T1 + alpha*T2
        current_sum = base_sd.get(key, torch.tensor(0.0, device=target_device, dtype=target_dtype)) \
                           .clone().to(device=target_device, dtype=target_dtype)

        if key in edited_task_vector1:
            current_sum += ALPHA_TASK_VECTOR_SCALE * edited_task_vector1[key].to(device=target_device, dtype=target_dtype)
        
        if key in edited_task_vector2:
            current_sum += ALPHA_TASK_VECTOR_SCALE * edited_task_vector2[key].to(device=target_device, dtype=target_dtype)
            
        final_merged_sd[key] = current_sum
    
    del edited_task_vector1, edited_task_vector2, base_sd
    torch.cuda.empty_cache()

    # 4. Save
    print("Loading final weights...")
    base_model.load_state_dict(final_merged_sd, assign=True, strict=False)
    
    print(f"Saving merged model to: {MERGED_MODEL_SAVE_PATH}")
    Path(MERGED_MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)

    print("CAT Merging Completed Successfully.")