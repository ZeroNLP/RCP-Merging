import pyarrow.parquet as pq
import pandas as pd
import json
import os
import numpy as np # Import numpy

# Function to convert numpy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert ndarray to list
    elif isinstance(obj, pd.Timestamp): # Handle pandas Timestamps if any
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

# 读取Parquet文件
parquet_path = "/data/jyyang/open-r1/medical/benchmark/mmlu/college_biology/test-00000-of-00001.parquet"
table = pq.read_table(parquet_path)
df = table.to_pandas()

# 随机抽取5条数据 (the original code said 10 but sampled 5, I'll stick to 5 as in df.sample)
# If you intended 10, change n=5 to n=10
sampled_data_raw = df.sample(n=5, random_state=42).to_dict(orient='records')

# Convert any numpy types in the sampled data
sampled_data = convert_numpy_types(sampled_data_raw)

# 创建输出目录（若不存在）
# Fixed the output_dir path to correctly point to the directory for reason_sample.json
output_json_path = "/data/jyyang/model_merge/EWC_Fisher/medical_sample/medical_sample_1.json"
output_dir = os.path.dirname(output_json_path)
os.makedirs(output_dir, exist_ok=True)

# 保存为JSON文件
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=4)

# Updated print statement to reflect the actual number of samples saved (5)
# and the correct output path variable.
print(f"已成功保存5条随机数据至 {output_json_path}")