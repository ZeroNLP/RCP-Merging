import os
import json
import torch
import copy
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

# --- 0. 配置相关路径和模型名称 ---
# 主模型路径
PATH_DEEPSEEK = "/data/transformers/DeepSeek-R1-Distill-Qwen-7B/"
PATH_MEDITRON = "/data/transformers/Meditron3-Qwen2.5-7B/"
PATH_QWEN = "/data/transformers/Qwen2.5-7B/"

# 输出路径
OUTPUT_DIR = "/data/jyyang/model_merge/mergekit/merged_models/two_models/Medical/EWC/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSON 样本文件位置
JSON_SAMPLES_PATH = "/data/jyyang/model_merge/EWC_Fisher/medical_sample/medical_samples.json"

# 超参数 (不再使用 LAMBDA_VAL, 但保留以便理解原始逻辑)
# LAMBDA_VAL = 1000.0

# 使用 bfloat16 配合 device_map="auto" 来节省内存，如果硬件不支持可换回 float32
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

# --- 辅助函数 ---
def flatten_model_params(model, target_device='cpu'):
    """将模型参数展平成一个单一向量，并放到目标CPU设备上"""
    # Parameters in a model loaded with device_map="auto" can be on different devices.
    # Move each to target_device (CPU) before concatenation.
    flat_params_on_target_device = [p.data.to(target_device, non_blocking=True).view(-1) for p in model.parameters()]
    return torch.cat(flat_params_on_target_device)

def unflatten_params_to_state_dict(flat_params_cpu, reference_model_auto_device):
    """
    将展平的CPU参数向量恢复为模型的state_dict。
    flat_params_cpu: 展平的参数，在CPU上。
    reference_model_auto_device: 参考模型，使用device_map="auto"加载，其参数在不同设备上。
    """
    new_state_dict = OrderedDict()
    current_pos = 0
    for name, param in reference_model_auto_device.named_parameters():
        num_elements = param.numel()
        # param.device 是此特定参数所在的设备 (由于 device_map="auto")
        # 从CPU张量中切片，然后移动到目标设备，再调整形状
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
    """加载模型 (device_map="auto") 并返回模型对象和展平到CPU的参数"""
    print(f"Loading model from {model_path} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=MODEL_DTYPE, # Using bfloat16 for memory efficiency
        device_map="auto",
        trust_remote_code=True
    )
    model.eval() # 设置为评估模式
    print(f"Model {model_path} loaded. Main device: {model.device}")
    with torch.no_grad():
        # Parameters will be moved to params_target_device (CPU) during flattening
        flat_params = flatten_model_params(model, target_device=params_target_device)
    print(f"Parameters for {model_path} flattened to {params_target_device}. Parameter count: {len(flat_params)}")
    return model, flat_params

# --- 1. 加载模型并将参数展平到CPU ---
print("--- Step 1: Loading models and flattening parameters to CPU ---")
# 使用 try-except-finally 确保即使出错也尝试释放模型占用的内存
model_deepseek = None
meditron_model_for_grads = None # Will be loaded specifically for grads
temp_qwen_model = None

try:
    # model_deepseek is kept in memory as its structure is needed for unflattening later
    model_deepseek, model_deepseek_flat_cpu = load_model_and_flatten_params(PATH_DEEPSEEK, params_target_device='cpu')
    model_deepseek_flat_original_cpu = model_deepseek_flat_cpu.clone() # This is already on CPU

    # For Meditron and Qwen, we only need their flattened CPU parameters for calculations initially.
    # The Meditron model object for gradients will be loaded separately or re-used if memory allows.
    # To save VRAM, we load, flatten to CPU, then delete the model object if it's not the one for gradients.

    # Load Meditron, flatten its params to CPU, then we can decide to keep or del the model object
    # For now, let's assume meditron_model_for_grads will be this one.
    meditron_model_for_grads, model_meditron_flat_cpu = load_model_and_flatten_params(PATH_MEDITRON, params_target_device='cpu')

    temp_qwen_model, model_qwen_flat_cpu = load_model_and_flatten_params(PATH_QWEN, params_target_device='cpu')
    N = len(model_deepseek_flat_cpu)
    del temp_qwen_model # Qwen model object no longer needed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"All models' parameters flattened to CPU. Parameter vector dimension N = {N}")

    if not (len(model_meditron_flat_cpu) == N and len(model_qwen_flat_cpu) == N):
        raise ValueError("Model parameter dimensions do not match! Ensure all models have the same architecture.")

    # Removed Fisher matrix loading, as it's no longer used.

    # --- 2. 计算 Sensitivity Matrices (每个样本一个N维CPU张量) ---
    print("\n--- Step 2: Calculating Sensitivity Matrices ---")
    with open(JSON_SAMPLES_PATH, "r", encoding="utf-8") as f:
        reason_samples = json.load(f)
    print(f"Loaded {len(reason_samples)} samples from {JSON_SAMPLES_PATH}.")

    # Instead of a list of raw sensitivity matrices, we'll sum them directly into one.
    total_sensitivity_matrix_cpu = torch.zeros(N, dtype=torch.float32, device='cpu')

    # meditron_model_for_grads is already loaded with device_map="auto"
    tokenizer = AutoTokenizer.from_pretrained(PATH_MEDITRON, trust_remote_code=True)

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

        # Accumulate sensitivities directly
        total_sensitivity_matrix_cpu += current_sample_sensitivity_flat_cpu
        print(f"  Sample {sample_idx+1}: Sensitivity map generated and accumulated (CPU).")
        del input_ids, labels, outputs, loss # Clean up GPU tensors for this sample
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


    del meditron_model_for_grads, original_meditron_state_dict_for_grads # Done with this model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Finished generating accumulated sensitivity matrix on CPU.")

    # --- 3. 根据 sensitivity_matrix_cpu 计算合并权重 ---
    print("\n--- Step 3: Calculating Merging Weights based on Sensitivity ---")

    # The 'importance_tensor_cpu' will now be derived directly from total_sensitivity_matrix_cpu
    # We want to merge weights where sensitivity_matrix_cpu > 0.
    # Create a boolean mask where sensitivity is positive.
    # This directly replaces the complex multi-GPU calculation.
    
    # We no longer need param_diff_sq_cpu or fisher_matrix_flat_cpu.
    
    # Create the indices for merging
    # `total_sensitivity_matrix_cpu` is the sum of sensitivities across all samples.
    # We want to merge where this *sum* is positive.
    update_indices_cpu = torch.where(total_sensitivity_matrix_cpu > 0)[0]

    print(f"  Number of parameters to update based on positive accumulated sensitivity: {len(update_indices_cpu)} out of {N}")

    # No need for intermediate multi-GPU steps here, as the condition is direct.

    # --- 4. 模型权重合并 (all on CPU) ---
    print("\n--- Step 4: Merging Model Weights (CPU) ---")

    model_diff_med_qwen_flat_cpu = model_meditron_flat_cpu - model_qwen_flat_cpu
    merged_model_flat_cpu = model_deepseek_flat_original_cpu.clone() # Start with DeepSeek original (CPU)

    # Apply the merge based on `update_indices_cpu`
    merged_model_flat_cpu[update_indices_cpu] += model_diff_med_qwen_flat_cpu[update_indices_cpu]
    print("Model weights merged on CPU.")

    # Clean up tensors that are no longer needed
    del model_diff_med_qwen_flat_cpu
    del total_sensitivity_matrix_cpu # This replaces importance_tensor_cpu in the new logic
    del update_indices_cpu
    # model_deepseek_flat_original_cpu, model_meditron_flat_cpu, model_qwen_flat_cpu can be cleared if no longer needed
    # merged_model_flat_cpu is the final result on CPU

    # --- 5. 保存合并后的模型和配置文件 ---
    print("\n--- Step 5: Saving Merged Model and Configuration ---")

    # model_deepseek (loaded with device_map="auto") is used as reference for structure and devices
    # merged_model_flat_cpu is on CPU
    print("Unflattening merged parameters back to state_dict with original device distribution...")
    merged_state_dict = unflatten_params_to_state_dict(merged_model_flat_cpu, model_deepseek)

    output_model_file = os.path.join(OUTPUT_DIR, "pytorch_model.bin") # HF usually expects this name
    torch.save(merged_state_dict, output_model_file)
    print(f"Merged model state_dict saved to {output_model_file}")

    # Copy config files from the DeepSeek model directory (or any base model)
    config_source_dir = PATH_DEEPSEEK
    files_to_copy = ["config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
    # Add any other files like generation_config.json if they exist and are needed
    optional_files = ["generation_config.json", "vocab.json", "merges.txt"] # For some tokenizers

    for filename in files_to_copy + optional_files:
        src_file = os.path.join(config_source_dir, filename)
        dst_file = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {filename} to {OUTPUT_DIR}")
        elif filename in files_to_copy: # Only warn if a mandatory file is missing
             print(f"Warning: Source file {src_file} not found, not copied.")

    print(f"\nProcess completed. Merged model and config files are in: {OUTPUT_DIR}")
    print(f"You should now be able to load the merged model using: AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}', device_map='auto', trust_remote_code=True)")

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
    if 'fisher_matrix_flat_cpu' in locals(): del fisher_matrix_flat_cpu # Although no longer used, keep for robustness if the var somehow exists
    if 'merged_model_flat_cpu' in locals(): del merged_model_flat_cpu

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleanup complete.")