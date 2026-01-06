import os
import torch
import shutil
from transformers import AutoModelForCausalLM
from collections import OrderedDict

# --- 0. 配置相关路径和模型名称 ---
# 基础模型路径
PATH_DEEPSEEK = "/data/transformers/DeepSeek-R1-Distill-Qwen-7B/"
# 专家模型路径 (其权重将在Fisher > 0的位置被合并)
PATH_MEDITRON = "/data/transformers/Meditron3-Qwen2.5-7B/"

# Fisher矩阵路径
FISHER_MATRIX_PATH = "/data/jyyang/model_merge/EWC_Fisher/Fisher_matrix/Fisher_matrix.pt"

# 输出路径
OUTPUT_DIR = "/data/jyyang/model_merge/mergekit/merged_models/two_models/Medical/EWC_only_penalty/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用 bfloat16 配合 device_map="auto" 来节省内存
MODEL_DTYPE = torch.bfloat16

print(f"使用模型数据类型: {MODEL_DTYPE}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")


# --- 辅助函数 ---
def flatten_model_params(model, target_device='cpu'):
    """将模型参数展平成一个单一向量，并移动到目标CPU设备上。"""
    flat_params = [p.data.to(target_device, non_blocking=True).view(-1) for p in model.parameters()]
    return torch.cat(flat_params)

def unflatten_params_to_state_dict(flat_params_cpu, reference_model_auto_device):
    """将展平的CPU参数向量恢复为模型的state_dict，并尊重原始设备分布。"""
    new_state_dict = OrderedDict()
    current_pos = 0
    for name, param in reference_model_auto_device.named_parameters():
        num_elements = param.numel()
        # 从CPU张量中切片，然后移动到参数所在的原始设备，再调整形状
        chunk_on_cpu = flat_params_cpu[current_pos : current_pos + num_elements]
        new_state_dict[name] = chunk_on_cpu.to(param.device, non_blocking=True).view_as(param.data).clone()
        current_pos += num_elements
    if current_pos != len(flat_params_cpu):
        raise ValueError("参数展平恢复时尺寸不匹配。")
    return new_state_dict

def load_model_and_flatten_params(model_path, params_target_device='cpu'):
    """加载模型 (device_map="auto") 并返回模型对象和展平到CPU的参数。"""
    print(f"从 {model_path} 加载模型 (device_map='auto')...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=MODEL_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"模型 {model_path} 已加载。")
    with torch.no_grad():
        flat_params = flatten_model_params(model, target_device=params_target_device)
    print(f"{model_path} 的参数已展平至 {params_target_device}。参数数量: {len(flat_params)}")
    return model, flat_params

# --- 主流程 ---
model_deepseek = None
model_meditron_temp = None
try:
    # --- 1. 加载模型和Fisher矩阵 ---
    print("\n--- 步骤 1: 加载模型和Fisher矩阵 ---")
    
    # 加载基础模型DeepSeek，保留其对象结构用于后续恢复
    model_deepseek, model_deepseek_flat_cpu = load_model_and_flatten_params(PATH_DEEPSEEK, params_target_device='cpu')
    
    # 加载专家模型Meditron，展平其参数后即可释放模型对象以节省显存
    model_meditron_temp, model_meditron_flat_cpu = load_model_and_flatten_params(PATH_MEDITRON, params_target_device='cpu')
    del model_meditron_temp # 释放Meditron模型对象
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    N = len(model_deepseek_flat_cpu)
    if len(model_meditron_flat_cpu) != N:
        raise ValueError("模型参数维度不匹配！请确保所有模型架构相同。")

    print(f"加载Fisher矩阵: {FISHER_MATRIX_PATH} 到CPU...")
    loaded_fisher_data = torch.load(FISHER_MATRIX_PATH, map_location='cpu')

    # 处理不同格式的Fisher矩阵文件
    if isinstance(loaded_fisher_data, torch.Tensor):
        fisher_matrix_flat_cpu = loaded_fisher_data.view(-1)
    elif isinstance(loaded_fisher_data, list) and len(loaded_fisher_data) > 0 and isinstance(loaded_fisher_data[0], torch.Tensor):
        fisher_matrix_flat_cpu = loaded_fisher_data[0].view(-1)
    else:
        raise TypeError(f"无法识别的Fisher矩阵文件格式: {type(loaded_fisher_data)}")

    if len(fisher_matrix_flat_cpu) != N:
        raise ValueError(f"Fisher矩阵维度 ({len(fisher_matrix_flat_cpu)}) 与模型参数维度 ({N}) 不匹配。")
    print("Fisher矩阵加载成功。")

    # --- 2. 根据Fisher矩阵合并模型权重 ---
    print("\n--- 步骤 2: 根据Fisher矩阵合并模型权重 ---")
    
    # 将基础模型参数作为合并的起点
    merged_model_flat_cpu = model_deepseek_flat_cpu.clone()

    # 使用GPU（如果可用）来加速索引计算
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备 {device} 来确定需要更新的参数索引...")

    fisher_on_device = fisher_matrix_flat_cpu.to(device)
    # 找到Fisher矩阵中大于0的元素的索引
    update_indices = torch.where(fisher_on_device > 0)[0]
    
    # 将索引移回CPU，以便在CPU张量上进行操作
    update_indices_cpu = update_indices.to('cpu')
    
    print(f"根据Fisher矩阵 (值 > 0), 将更新 {len(update_indices_cpu)} / {N} 个参数。")
    
    # 释放GPU上的Fisher矩阵以节省显存
    del fisher_on_device, update_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 在CPU上执行合并操作
    # 在Fisher > 0的索引位置，使用Meditron模型的权重替换DeepSeek的权重
    print("正在合并权重：在指定位置使用Meditron的权重...")
    merged_model_flat_cpu[update_indices_cpu] = model_meditron_flat_cpu[update_indices_cpu]
    
    print("权重合并完成。")
    # 清理不再需要的大型CPU张量
    del model_deepseek_flat_cpu, model_meditron_flat_cpu, fisher_matrix_flat_cpu, update_indices_cpu

    # --- 3. 保存合并后的模型 ---
    print("\n--- 步骤 3: 保存合并后的模型和配置文件 ---")
    
    print("将合并后的参数恢复为模型的 state_dict...")
    # 使用原始DeepSeek模型的结构和设备分布来恢复
    merged_state_dict = unflatten_params_to_state_dict(merged_model_flat_cpu, model_deepseek)
    
    # 保存模型权重
    output_model_file = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    torch.save(merged_state_dict, output_model_file)
    print(f"合并后的模型 state_dict 已保存至: {output_model_file}")

    # 从基础模型目录复制配置文件
    config_source_dir = PATH_DEEPSEEK
    files_to_copy = ["config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "generation_config.json"]
    optional_files = ["vocab.json", "merges.txt"] 

    for filename in files_to_copy + optional_files:
        src_file = os.path.join(config_source_dir, filename)
        dst_file = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"已复制 {filename} 到 {OUTPUT_DIR}")
        elif filename in files_to_copy:
             print(f"警告: 必需的配置文件 {src_file} 未找到，无法复制。")
            
    print(f"\n 处理完成。合并后的模型位于: {OUTPUT_DIR}")
    print(f"现在你可以使用以下代码加载新模型: AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}', device_map='auto', trust_remote_code=True)")

except Exception as e:
    print(f"\n 处理过程中发生错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n正在清理资源...")
    # 确保所有模型对象和大型张量都被删除
    del model_deepseek
    if 'model_deepseek_flat_cpu' in locals(): del model_deepseek_flat_cpu
    if 'model_meditron_flat_cpu' in locals(): del model_meditron_flat_cpu
    if 'fisher_matrix_flat_cpu' in locals(): del fisher_matrix_flat_cpu
    if 'merged_model_flat_cpu' in locals(): del merged_model_flat_cpu
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("清理完成。")