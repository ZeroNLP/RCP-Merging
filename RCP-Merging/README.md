# Fisher Matrix Calculation Tool

This repository contains tools to calculate the **Fisher Information Matrix (FIM)** for a specific Large Language Model (LLM) based on a reasoning dataset. The FIM is often used to estimate parameter importance for tasks such as model merging, pruning, or continual learning.

## Files Description

1.  **`run_fisher.sh`**: A Bash script that serves as the entry point. It contains all configurable hyperparameters and executes the Python script.
2.  **`calculate_fisher.py`**: The core Python script that loads the model, processes the data, computes gradients, and saves the averaged Fisher Matrix.

## Usage

### 1. Prerequisites
Ensure you have the necessary Python libraries installed:
```bash
pip install torch transformers
```

# RCP-Merging Tool

This project implements a model merging strategy that combines a base model (DeepSeek) with a domain-specific model (Meditron) using a common ancestor (Qwen) as a baseline. It utilizes Fisher Information Matrices and sensitivity analysis on medical samples to determine which parameters to update.

## Prerequisites

- Python 3.8+
- PyTorch (with CUDA support)
- Transformers
- A machine with one or (ideally) multiple GPUs.

## Files Structure

- `run_merge.sh`: The entry point script. **Edit this file to set your paths and hyperparameters.**
- `merge_models.py`: The logic script performing the calculations and merging.

## How to Use

1. **Configure Paths**: Open `run_merge.sh` and update the path variables to point to your local model directories and data files:
    - `DEEPSEEK_PATH`: The target model you want to improve.
    - `MEDITRON_PATH`: The source of domain knowledge.
    - `QWEN_PATH`: The common base model used for vector subtraction.
    - `FISHER_MATRIX_PATH`: Path to the pre-calculated Fisher matrix (`.pt` file).
    - `JSON_SAMPLES_PATH`: Path to the JSON file containing medical QA samples.

2. **Set Hyperparameters**:
    - Adjust `LAMBDA_VAL` in `run_merge.sh`. This controls the weight of the Fisher information vs. the sensitivity analysis.

3. **Run the Script**:
    ```bash
    bash run_merge.sh
    ```

## Parameter Explanation

The script calculates an **Importance Tensor** to decide which parameters in the DeepSeek model should be updated with the weights from Meditron.

The calculation logic typically follows:
$$ \text{Score} = (\text{Sensitivity} + \lambda \times (\text{Fisher} \times \text{ParamDiff}^2)) $$

- **Sensitivity**: Calculated dynamically by running gradients on the provided medical samples (`JSON_SAMPLES_PATH`).
- **Fisher**: The Fisher Information Matrix representing parameter importance in the general domain.
- **ParamDiff**: The squared difference between DeepSeek and Meditron weights.
- **LAMBDA_VAL** ($\lambda$): A scaling factor.
  - A higher `LAMBDA_VAL` places more emphasis on the Fisher matrix (preserving general capability).
  - A lower `LAMBDA_VAL` allows the medical sensitivity (domain capability) to have more influence.

## Multi-GPU Logic

The script attempts to distribute the heavy tensor operations across up to 3 GPUs to manage memory usage:
- **GPU 0**: Calculates the product of Fisher Matrix and Parameter Difference.
- **GPU 1**: Scales the result by `LAMBDA_VAL`.
- **GPU 2**: Adds the sensitivity matrix to the scaled product.