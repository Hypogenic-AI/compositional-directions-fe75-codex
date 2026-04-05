# Downloaded Datasets

This directory contains datasets for studying compositionality of linear directions and activation steering in LLM residual streams. Data files are not intended for git commits; see `.gitignore`.

## Dataset 1: TruthfulQA (multiple-choice validation)

### Overview
- Source: `truthful_qa` (Hugging Face datasets)
- Size: 817 examples
- Format: Hugging Face Dataset saved to disk
- Task: Truthfulness / misconception resistance (MCQA)
- Splits: validation
- License: Check dataset card on Hugging Face

### Download Instructions

Using Hugging Face (recommended):
```python
from datasets import load_dataset

ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
ds.save_to_disk("datasets/truthful_qa_multiple_choice_validation")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/truthful_qa_multiple_choice_validation")
print(len(ds), ds.column_names)
```

### Sample Data
- Saved sample: `datasets/truthful_qa_multiple_choice_validation/samples/samples.json`

### Notes
- Useful for testing whether truthfulness-related steering vectors transfer across prompts.
- Can be used for evaluating direction consistency and side effects.

## Dataset 2: IFEval (train)

### Overview
- Source: `google/IFEval` (Hugging Face datasets)
- Size: 541 examples
- Format: Hugging Face Dataset saved to disk
- Task: Instruction-following constraints (format/length/keywords)
- Splits: train
- License: Check dataset card on Hugging Face

### Download Instructions
```python
from datasets import load_dataset

ds = load_dataset("google/IFEval", split="train")
ds.save_to_disk("datasets/google_ifeval_train")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/google_ifeval_train")
print(len(ds), ds.column_names)
```

### Sample Data
- Saved sample: `datasets/google_ifeval_train/samples/samples.json`

### Notes
- Strong fit for multi-property composition experiments.
- Supports evaluating whether summed directions satisfy multiple constraints simultaneously.

## Dataset 3: HellaSwag (validation)

### Overview
- Source: `Rowan/hellaswag` (Hugging Face datasets)
- Size: 10,042 examples
- Format: Hugging Face Dataset saved to disk
- Task: Commonsense completion / robustness check
- Splits: validation
- License: Check dataset card on Hugging Face

### Download Instructions
```python
from datasets import load_dataset

ds = load_dataset("Rowan/hellaswag", split="validation")
ds.save_to_disk("datasets/rowan_hellaswag_validation")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/rowan_hellaswag_validation")
print(len(ds), ds.column_names)
```

### Sample Data
- Saved sample: `datasets/rowan_hellaswag_validation/samples/samples.json`

### Notes
- Useful as an off-target capability retention benchmark after steering.
- Helps detect regressions when applying steering vectors.

## Validation Summary

Quick checks performed:
- Loaded each saved dataset from disk successfully.
- Verified row counts and schema.
- Saved first 10 records for each dataset in `samples/samples.json`.

Machine-readable summary: `datasets/dataset_summary.json`
