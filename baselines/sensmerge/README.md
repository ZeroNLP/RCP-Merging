# Sens-Merging Tool

This project implements the "Sens-Merging" algorithm, which merges two Fine-Tuned (SFT) models into a single base model. It calculates task-specific sensitivities (Alpha) and cross-task alignment scores (Tau) to derive optimal merging coefficients for every layer.

## Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Transformers
- tqdm

Install dependencies via pip:
```bash
pip install torch transformers tqdm