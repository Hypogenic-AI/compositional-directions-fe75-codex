# Cloned Repositories

## Repo 1: function_vectors
- URL: https://github.com/ericwtodd/function_vectors
- Purpose: Official implementation for Function Vectors in LLMs (ICLR 2024)
- Location: `code/function_vectors/`
- Commit: `751e221`
- Key files:
  - `notebooks/fv_demo.ipynb`
  - `src/eval_scripts/`
  - `src/utils/extract_utils.py`
  - `dataset_files/`
- Notes:
  - Includes code and datasets used in the paper.
  - Main utilities for extracting function vectors and intervention-time evaluation.
  - Environment is defined in `fv_environment.yml` (conda-oriented; can be translated to pip/uv if needed).

## Repo 2: llm-steer-instruct
- URL: https://github.com/microsoft/llm-steer-instruct
- Purpose: Activation steering for improved instruction following (ICLR 2025)
- Location: `code/llm-steer-instruct/`
- Commit: `9dac937`
- Key files:
  - `format/compute_representations.py`
  - `format/find_best_layer.py`
  - `length/evaluate.py`
  - `keywords/evaluate.py`
  - `composition/evaluate_format_plus_length.py`
  - `config/`
- Notes:
  - Structured by experiment type: format, length, keyword, and multi-instruction composition.
  - Uses Hydra config system and IFEval scripts.
  - Good reference for compositional steering experiments on instruction constraints.

## Repo 3: activation_steering
- URL: https://github.com/cma1114/activation_steering
- Purpose: Experimental activation steering toolkit and analyses
- Location: `code/activation_steering/`
- Commit: `5a57c6a`
- Key files:
  - `enhanced_hooking.py`
  - `steering-honesty.ipynb`
  - `plot_steering_scores.ipynb`
- Notes:
  - Practical exploration of behavior vectors and steering robustness.
  - Useful for prompt-pair construction and behavior-vector diagnostics.

## Repo 4: pyvene
- URL: https://github.com/stanfordnlp/pyvene
- Purpose: General intervention framework for internal model states
- Location: `code/pyvene/`
- Commit: `9e33390`
- Key files:
  - `pyvene/` package
  - docs at https://stanfordnlp.github.io/pyvene/
- Notes:
  - Lightweight reusable framework for activation interventions.
  - Useful for implementing controlled residual-stream interventions without bespoke hooks.

## Repo 5: TransformerLens
- URL: https://github.com/TransformerLensOrg/TransformerLens
- Purpose: Mechanistic interpretability toolkit for transformer internals
- Location: `code/TransformerLens/`
- Commit: `9c5a2a8`
- Key files:
  - `transformer_lens/` package
  - `demos/`
- Notes:
  - Strong default library for caching/editing residual-stream activations.
  - Useful for probing linear directions, layerwise subspaces, and ablation experiments.

## Quick Validation Performed
- Confirmed repositories cloned successfully.
- Read each top-level README to identify scope and entry points.
- Captured commit hashes for reproducibility.

## Suggested Immediate Reuse
1. Use `function_vectors` to replicate baseline FV extraction and composition checks.
2. Use `llm-steer-instruct` for multi-property composition evaluation protocol.
3. Use `TransformerLens` or `pyvene` to implement controlled linear-direction interventions in a unified code path.
