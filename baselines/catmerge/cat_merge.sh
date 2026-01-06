#!/bin/bash

# ==========================================
# CAT Merging Configuration & Execution Script
# ==========================================

# --- File Paths ---
# Path to the base pre-trained model (e.g., Llama-3)
export BASE_MODEL_PATH="/data/transformers/Meta-Llama-3.1-8B"

# Path and Name for Model 1 (Reasoning Expert)
export MODEL_1_PATH="/data/transformers/DeepSeek-R1-Distill-Llama-8B"
export MODEL_1_NAME="deepseek_reasoning"

# Path and Name for Model 2 (Medical Expert)
export MODEL_2_PATH="/data/transformers/Llama3-OpenBioLLM-8B"
export MODEL_2_NAME="Llama3-OpenBioLLM-8B"

# Output path for the final merged model
export MERGED_MODEL_SAVE_PATH="./output/merged_reasoning_medical_llama3"

# Paths to calibration data (JSON files)
export REASONING_SAMPLES_PATH="../../RCP-Merging/calibration/reason_sample/reason_sample.json"
export MEDICAL_SAMPLES_PATH="../../RCP-Merging/calibration/finance_sample/finance_sample.json"

# --- Merging Hyperparameters ---
# Lambda: Coefficient for penalizing conflicting features (Higher = more strict conflict removal)
export LAMBDA_CONFLICT_PRESERVE=1.0

# C Dim Trim: Number of dimensions to trim in the projection basis
export C_DIM_TRIM=3

# Alpha: Scaling factor for the task vectors (1.0 = full strength)
export ALPHA_TASK_VECTOR_SCALE=1.0

# Number of exemplars to use for feature extraction (Covariance calculation)
export NUM_EXEMPLARS_PER_TASK=5

# --- System & Computation Settings ---
# System prompt injected during reasoning sample tokenization
export SYS_PROMPT_REASONING="You are a helpful AI assistant. Provide clear and concise answers."

# PyTorch Device Configuration (matches Python script logic)
# 'auto' allows transformers to handle allocation; specific indices can be set via CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1

# ==========================================
# Execution
# ==========================================
echo "Starting CAT Merging Process..."
echo "Base Model: $BASE_MODEL_PATH"
echo "Output Dir: $MERGED_MODEL_SAVE_PATH"

# Run the Python script
python cat_merge.py