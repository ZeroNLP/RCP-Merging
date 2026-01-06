# Baselines

This directory contains various baseline methods for merging models in the RCP-Merging project. Each subfolder represents a specific merging approach or utility. Below is a detailed description of each subfolder and instructions for using them.

---

## `catmerge`

### Description
`catmerge` implements the "Context-Aware Task Merging" (CAT Merging) approach. This method merges two expert models (e.g., Reasoning and Medical) into a base model using Task Vector Trimming.

### Contents
- `cat_merge.py`: Python script for executing the CAT Merging algorithm.
- `cat_merge.sh`: Bash script for running the merging process.
- `README.md`: Documentation for CAT Merging.

### Usage
1. Ensure the required environment variables are set. These include paths to the base model, expert models, and other hyperparameters.
2. Make the Bash script executable:
   ```bash
   chmod +x cat_merge.sh
   ```
3. Run the script:
   ```bash
   ./cat_merge.sh
   ```
4. The merged model will be saved to the specified output directory.

---

## `examples`

### Description
This folder contains example configuration files for different merging methods. These YAML files define the models, parameters, and merging strategies.

### Contents
- `dare_linear.yml`: Configuration for the `dare_linear` merging method.
- `dare_ties.yml`: Configuration for the `dare_ties` merging method.
- `fusellm.yml`: Configuration for the `fusellm` merging method.
- `linear.yml`: Configuration for the `linear` merging method.
- `task_arithmetic.yml`: Configuration for the `task_arithmetic` merging method.
- `ties.yml`: Configuration for the `ties` merging method.

### Usage
1. Select the appropriate YAML file for your merging method.
2. Modify the file to specify the paths to your models and desired parameters.
3. Use the configuration file with the corresponding merging script.

---

## `mergekit`

### Description
`mergekit` provides a comprehensive toolkit for model merging. It includes utilities for data processing, architecture manipulation, and merging strategies.

### Contents
- `merge.py`: Core script for merging models.
- `config.py`: Configuration utilities.
- `common.py`: Common helper functions.
- `scripts/`: Additional scripts for specific tasks.
- `tokenizer/`: Tokenizer utilities.
- `_data/`, `io/`, `architecture/`, `merge_methods/`, `moe/`: Submodules for various functionalities.

### Usage
1. Install the required dependencies.
2. Refer to `merge.py` for the main merging logic.
3. Use the provided scripts and utilities as needed for your specific merging tasks.

---

## `sensmerge`

### Description
`sensmerge` implements the "Sens-Merging" algorithm. This method merges two fine-tuned models into a single base model by calculating task-specific sensitivities (Alpha) and cross-task alignment scores (Tau) to derive optimal merging coefficients for each layer.

### Contents
- `sens_merging.py`: Python script for executing the Sens-Merging algorithm.
- `sens_merging.sh`: Bash script for running the merging process.
- `README.md`: Documentation for Sens-Merging.

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)
- Transformers
- tqdm

Install dependencies:
```bash
pip install torch transformers tqdm
```

### Usage
1. Make the Bash script executable:
   ```bash
   chmod +x sens_merging.sh
   ```
2. Run the script:
   ```bash
   ./sens_merging.sh
   ```
3. The merged model will be saved to the specified output directory.

---

## General Notes
- Ensure all dependencies are installed before running any scripts.
- Modify configuration files as needed to suit your specific models and tasks.
- Refer to the individual `README.md` files in each subfolder for more details.