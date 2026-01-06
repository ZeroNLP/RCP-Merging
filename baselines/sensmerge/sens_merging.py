import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import argparse
from collections import OrderedDict

# --- Utility Functions ---

def load_json_dataset(path):
    """Loads a JSON dataset from the given path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return []

def get_medical_input(sample):
    """Extracts input from a medical dataset sample."""
    return sample.get("question", "")

def get_reasoning_input(sample, sys_prompt):
    """Constructs input prompt for the reasoning dataset."""
    input_reannotated_assistant_content = sample.get("reannotated_assistant_content", "")
    input_problem = sample.get("problem", "")
    input_solution = sample.get("solution", "")

    input_prompt = (
        f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\nProblem:\n{input_problem}\nModel Answer:\n{input_reannotated_assistant_content}\nSolution:\n{input_solution}\n"
        f"[/INST]"
    )
    return input_prompt

# --- Input Processors ---
# Define processors explicitly to make their names stable and dispatch logic robust.

def medical_input_processor(sample):
    """Processes medical input. Intended for data_processing_fns."""
    return get_medical_input(sample)

def reasoning_input_processor(sample, sys_prompt_override=None):
    """
    Processes reasoning input. Intended for data_processing_fns.
    Uses the global/passed system prompt if provided.
    """
    # Note: The actual system prompt is passed via the lambda or partial in the main loop or handled in dispatch
    return get_reasoning_input(sample, sys_prompt_override)

def load_model_and_tokenizer(model_path):
    """Loads a model and tokenizer from the given path using device_map='auto'."""
    print(f"Loading model and tokenizer from: {model_path} with device_map='auto'")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",          # Automatically distribute across GPUs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval() # Set to evaluation mode
        print(f"Model {model_path} loaded successfully. First param device: {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        raise

# --- Sens-Merging Algorithm Components ---

def calculate_loss_and_gradients(model, tokenizer, text_batch, target_device):
    """
    Calculates loss and gradients for a batch of text.
    Inputs are moved to target_device (e.g., primary GPU).
    """
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(target_device)
    labels = inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    model.zero_grad()
    loss.backward()

    return loss.item()

def get_logits(model, tokenizer, text_batch, target_device):
    """
    Gets model logits for a batch of text.
    """
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(target_device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits

def calculate_task_specific_scaling_factors(model_sft, tokenizer_sft, calibration_data, data_processing_fn, primary_device, num_samples, sys_prompt):
    """
    Calculates task-specific scaling factors (alpha_i^l) for each layer.
    """
    print(f"Calculating task-specific scaling factors for a model (using {data_processing_fn.__name__})...")
    layer_sensitivities_sum = OrderedDict()
    param_names_per_layer = OrderedDict()

    for name, param in model_sft.named_parameters():
        if not param.requires_grad:
            continue
        layer_name = ".".join(name.split(".")[:3]) # Group parameters by layer
        if layer_name not in layer_sensitivities_sum:
            layer_sensitivities_sum[layer_name] = 0.0
            param_names_per_layer[layer_name] = []
        param_names_per_layer[layer_name].append(name)

    num_processed = 0
    for i in tqdm(range(min(len(calibration_data), num_samples)), desc=f"Calibrating task-specific sensitivity ({data_processing_fn.__name__})"):
        sample = calibration_data[i]

        # Dispatch logic for calling the data processing function
        if "reasoning" in data_processing_fn.__name__:
            input_text = data_processing_fn(sample, sys_prompt)
        else:
            input_text = data_processing_fn(sample)

        if not input_text: 
            continue

        calculate_loss_and_gradients(model_sft, tokenizer_sft, [input_text], primary_device)
        num_processed += 1

        for layer_id, param_names in param_names_per_layer.items():
            current_layer_sensitivity_for_sample = 0.0
            for name in param_names:
                param_obj = dict(model_sft.named_parameters())[name]
                if param_obj.grad is not None:
                    sensitivity_val_tensor = torch.sum(torch.abs(param_obj.data * param_obj.grad.data))
                    current_layer_sensitivity_for_sample += sensitivity_val_tensor.item()
            layer_sensitivities_sum[layer_id] += current_layer_sensitivity_for_sample

    if num_processed == 0:
        raise ValueError(f"No calibration samples were processed for task-specific sensitivity using {data_processing_fn.__name__}.")

    raw_sensitivities_list = [s / num_processed for s in layer_sensitivities_sum.values()]
    raw_sensitivities = torch.tensor(raw_sensitivities_list, device=primary_device, dtype=torch.float32)

    norm_val = torch.norm(raw_sensitivities, p=2)
    if norm_val == 0:
        print(f"Warning: Norm of raw sensitivities is zero for {data_processing_fn.__name__}. Returning uniform scaling factors.")
        alpha_factors = torch.ones_like(raw_sensitivities) / len(raw_sensitivities) if len(raw_sensitivities) > 0 else torch.tensor([])
    else:
        alpha_factors = raw_sensitivities / norm_val

    alpha_map = OrderedDict(zip(layer_sensitivities_sum.keys(), alpha_factors.cpu().tolist()))
    print(f"Task-specific scaling factors (alpha) calculated using {data_processing_fn.__name__}.")
    return alpha_map, param_names_per_layer


def calculate_cross_task_scaling_factors(models_sft, tokenizers_sft, calibration_datasets, data_processing_fns_list, primary_device, num_samples, sys_prompt):
    """
    Calculates cross-task scaling factors (tau_i).
    """
    print("Calculating cross-task scaling factors (tau)...")
    num_models = len(models_sft)
    alignment_scores_g_ij_list = [[0.0 for _ in range(num_models)] for _ in range(num_models)]

    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue

            model_i = models_sft[i]
            tokenizer_i = tokenizers_sft[i]

            model_j_expert = models_sft[j]
            tokenizer_j_expert = tokenizers_sft[j]

            cal_data_j = calibration_datasets[j]
            proc_fn_j = data_processing_fns_list[j] 

            current_g_ij_sum = 0.0
            num_processed = 0

            for k in tqdm(range(min(len(cal_data_j), num_samples)), desc=f"Cross-task align: M{i+1} on Task{j+1} data"):
                sample = cal_data_j[k]

                if "reasoning" in proc_fn_j.__name__: 
                    input_text = proc_fn_j(sample, sys_prompt)
                else:
                    input_text = proc_fn_j(sample)

                if not input_text:
                    continue

                logits_i = get_logits(model_i, tokenizer_i, [input_text], primary_device)
                logits_j_expert = get_logits(model_j_expert, tokenizer_j_expert, [input_text], primary_device)

                logits_i = logits_i.to(primary_device, dtype=torch.float32)
                logits_j_expert = logits_j_expert.to(primary_device, dtype=torch.float32)

                min_seq_len = min(logits_i.shape[1], logits_j_expert.shape[1])

                l2_distance_tensor = torch.norm(logits_i[:, :min_seq_len, :] - logits_j_expert[:, :min_seq_len, :], p=2, dim=-1).mean()
                current_g_ij_sum += l2_distance_tensor.item()
                num_processed +=1

            if num_processed > 0:
                alignment_scores_g_ij_list[i][j] = current_g_ij_sum / num_processed
            else:
                print(f"Warning: No samples processed for g_({i},{j}).")
                alignment_scores_g_ij_list[i][j] = 0.0

    alignment_scores_g_ij = torch.tensor(alignment_scores_g_ij_list, device=primary_device, dtype=torch.float32)
    tau_raw = torch.sum(alignment_scores_g_ij, dim=1)

    norm_val_tau = torch.norm(tau_raw, p=1)
    if norm_val_tau == 0:
        print("Warning: Norm of raw tau is zero. Returning uniform cross-task factors.")
        tau_factors = torch.ones_like(tau_raw) / len(tau_raw) if len(tau_raw) > 0 else torch.tensor([])
    else:
        tau_factors = tau_raw / norm_val_tau

    print("Cross-task scaling factors (tau) calculated:", tau_factors.cpu().tolist())
    return tau_factors


def calculate_final_scaling_coefficients(alpha_maps, tau_factors, temperature, primary_device):
    """
    Calculates final scaling coefficients (sigma_i^l).
    """
    print("Calculating final scaling coefficients (sigma)...")
    num_models = len(alpha_maps)
    if num_models == 0 or not alpha_maps[0]:
        return []

    # Assuming all alpha_maps have the same layer structure
    first_valid_map_idx = -1
    for idx, amap in enumerate(alpha_maps):
        if amap:
            first_valid_map_idx = idx
            break
    
    if first_valid_map_idx == -1:
        return [OrderedDict() for _ in range(num_models)]

    layer_names = list(alpha_maps[first_valid_map_idx].keys())
    num_layers = len(layer_names)

    sigma_il_unnormalized = torch.zeros((num_models, num_layers), device=primary_device, dtype=torch.float32)

    for i in range(num_models):
        if not alpha_maps[i] or len(alpha_maps[i]) != num_layers: 
            print(f"Warning: alpha_map for model {i} is invalid. Using zeros.")
            current_alpha_values = torch.zeros(num_layers, device=primary_device, dtype=torch.float32)
        else:
            current_alpha_values_list = [alpha_maps[i].get(ln, 0.0) for ln in layer_names]
            current_alpha_values = torch.tensor(current_alpha_values_list, device=primary_device, dtype=torch.float32)

        if i < len(tau_factors):
            sigma_il_unnormalized[i, :] = current_alpha_values * tau_factors[i]
        else:
            sigma_il_unnormalized[i, :] = torch.zeros(num_layers, device=primary_device, dtype=torch.float32)

    sigma_il = F.softmax(sigma_il_unnormalized / temperature, dim=0)

    final_sigma_maps = []
    for i in range(num_models):
        final_sigma_maps.append(OrderedDict(zip(layer_names, sigma_il[i, :].cpu().tolist())))

    print("Final scaling coefficients (sigma) calculated.")
    return final_sigma_maps


def merge_models(base_model_sd, sft_model_sds, final_sigma_maps, param_names_per_layer_list, K):
    """
    Merges models based on final scaling coefficients.
    """
    print("Merging models...")
    merged_sd = OrderedDict()
    
    if not param_names_per_layer_list or not param_names_per_layer_list[0]:
        if base_model_sd: return base_model_sd.copy()
        return merged_sd 

    param_names_map_representative = param_names_per_layer_list[0]

    for layer_name, param_names_in_layer in tqdm(param_names_map_representative.items(), desc="Merging layers"):
        for param_name in param_names_in_layer:
            if param_name not in base_model_sd:
                # If parameter missing from base, try copying from first SFT model
                if K > 0 and param_name in sft_model_sds[0]:
                    merged_sd[param_name] = sft_model_sds[0][param_name].clone()
                continue

            base_param = base_model_sd[param_name]
            delta_sum = torch.zeros_like(base_param) 

            for i in range(K): 
                if i >= len(sft_model_sds) or param_name not in sft_model_sds[i]:
                    continue
                if i >= len(final_sigma_maps) or layer_name not in final_sigma_maps[i]:
                    sigma_i_l = 0.0
                else:
                    sigma_i_l = final_sigma_maps[i][layer_name]

                sft_param = sft_model_sds[i][param_name]
                task_vector_param = sft_param.to(base_param.device) - base_param 
                delta_sum += K * sigma_i_l * task_vector_param

            merged_sd[param_name] = base_param + delta_sum

    # Add remaining base model params (embeddings, etc.)
    all_merged_params_set = set(merged_sd.keys())
    for p_name, p_val in tqdm(base_model_sd.items(), desc="Adding remaining base model params"):
        if p_name not in all_merged_params_set:
            merged_sd[p_name] = p_val.clone()

    print("Models merged successfully.")
    return merged_sd

# --- Main Execution ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sens-Merging: Merge LLMs based on task sensitivity.")
    
    # Model Paths
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to the first SFT model (Medical)")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to the second SFT model (Reasoning)")
    
    # Dataset Paths
    parser.add_argument("--medical_dataset_path", type=str, required=True, help="Path to the medical calibration dataset (JSON)")
    parser.add_argument("--reasoning_dataset_path", type=str, required=True, help="Path to the reasoning calibration dataset (JSON)")
    
    # Output
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the merged model")
    
    # Hyperparameters
    parser.add_argument("--num_calibration_samples", type=int, default=100, help="Number of samples to use for calibration")
    parser.add_argument("--softmax_temperature", type=float, default=1.0, help="Temperature for Softmax scaling")
    parser.add_argument("--sys_prompt_reasoning", type=str, default="You are a helpful assistant.", help="System prompt for reasoning tasks")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Sens-Merging process... Using primary device: {DEVICE}")
    os.makedirs(args.output_path, exist_ok=True)

    # Load Data
    medical_data = load_json_dataset(args.medical_dataset_path)
    reasoning_data = load_json_dataset(args.reasoning_dataset_path)

    if not medical_data or not reasoning_data:
        print("Error: One or both datasets failed to load. Exiting.")
        return

    all_data_processing_fns = [medical_input_processor, reasoning_input_processor]
    calibration_datasets = [medical_data, reasoning_data]

    # Load Models
    print("Loading base model...")
    base_model, _ = load_model_and_tokenizer(args.base_model_path)
    print("Loading SFT model 1 (Medical)...")
    model1_sft, tokenizer1_sft = load_model_and_tokenizer(args.model1_path)
    print("Loading SFT model 2 (Reasoning)...")
    model2_sft, tokenizer2_sft = load_model_and_tokenizer(args.model2_path)

    sft_models_list = [model1_sft, model2_sft]
    sft_tokenizers_list = [tokenizer1_sft, tokenizer2_sft]

    # Calculate Alpha (Task Specific)
    alpha_map1, param_names_per_layer1 = calculate_task_specific_scaling_factors(
        model1_sft, tokenizer1_sft, medical_data, medical_input_processor, DEVICE, 
        args.num_calibration_samples, args.sys_prompt_reasoning
    )
    alpha_map2, param_names_per_layer2 = calculate_task_specific_scaling_factors(
        model2_sft, tokenizer2_sft, reasoning_data, reasoning_input_processor, DEVICE, 
        args.num_calibration_samples, args.sys_prompt_reasoning
    )
    alpha_maps = [alpha_map1, alpha_map2]
    param_names_per_layer_list_for_merge = [param_names_per_layer1, param_names_per_layer2]

    # Calculate Tau (Cross Task)
    tau_factors = calculate_cross_task_scaling_factors(
        sft_models_list, sft_tokenizers_list, calibration_datasets, all_data_processing_fns, DEVICE,
        args.num_calibration_samples, args.sys_prompt_reasoning
    )

    # Calculate Sigma (Final Coefficients)
    final_sigma_maps = calculate_final_scaling_coefficients(alpha_maps, tau_factors, args.softmax_temperature, DEVICE)
    if not final_sigma_maps or not any(final_sigma_maps):
        print("Error: Invalid final scaling coefficients. Exiting.")
        return

    # Prepare for Merge (Memory Management)
    base_model_sd_orig_device = base_model.state_dict()
    sft_model_sds_orig_device = [model1_sft.state_dict(), model2_sft.state_dict()]

    del base_model, model1_sft, model2_sft, sft_models_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models deleted from memory to free GPU resources before merging.")

    # Execute Merge
    merged_model_sd = merge_models(
        base_model_sd_orig_device,
        sft_model_sds_orig_device,
        final_sigma_maps,
        param_names_per_layer_list_for_merge,
        K=len(sft_model_sds_orig_device)
    )

    # Save Model
    print(f"Saving merged model to {args.output_path}...")
    merged_model_for_saving = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    merged_model_sd_cpu = OrderedDict((k, v.cpu()) for k, v in merged_model_sd.items())
    merged_model_for_saving.load_state_dict(merged_model_sd_cpu)
    merged_model_for_saving.save_pretrained(args.output_path)

    # Save Tokenizer
    tokenizer_to_save = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer_to_save.save_pretrained(args.output_path)

    print(f"Merged model saved to {args.output_path}")
    print("Sens-Merging process complete.")

if __name__ == "__main__":
    main()