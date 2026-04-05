## Resources Catalog

### Summary
This document catalogs all resources gathered for the research project on compositional linear directions in transformer residual streams.

### Papers
Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|---|---|---:|---|---|
| Function Vectors in Large Language Models | Todd et al. | 2023 | papers/2310.15213_function_vectors_in_large_language_models.pdf | Function vectors, causal extraction, partial composition |
| Steering Llama 2 via Contrastive Activation Addition | Panickssery et al. | 2023 | papers/2312.06681_steering_llama_2_via_contrastive_activation_addition.pdf | CAA residual-stream steering baseline |
| Steering Language Models With Activation Engineering | Turner et al. | 2023 | papers/2308.10248_steering_language_models_with_activation_engineering.pdf | ActAdd steering baseline |
| Improving Instruction-Following in Language Models through Activation Steering | Stolfo et al. | 2024 | papers/2410.12877_improving_instruction_following_in_language_models_through_a.pdf | Instruction vectors + composition |
| HyperSteer: Activation Steering at Scale with Hypernetworks | Sun et al. | 2025 | papers/2506.03292_hypersteer_activation_steering_at_scale_with_hypernetworks.pdf | Learned vector generator baseline |
| Steering Vector Fields for Context-Aware Inference-Time Control in LLMs | Li et al. | 2026 | papers/2602.01654_steering_vector_fields_for_context_aware_inference_time_cont.pdf | Context-dependent vector fields |
| Transformers learn factored representations | Shai et al. | 2026 | papers/2602.02385_transformers_learn_factored_representations.pdf | Factorized/orthogonal representation hypothesis |
| Progress measures for grokking via mechanistic interpretability | Nanda et al. | 2023 | papers/2301.05217_progress_measures_for_grokking_via_mechanistic_interpretabil.pdf | Mech-interp progress metrics |
| Multi-property Steering of LLMs with Dynamic Activation Composition | Scalena et al. | 2024 | papers/2406.17563_multi_property_steering_of_large_language_models_with_dynami.pdf | Dynamic multi-vector composition |

See `papers/README.md` for detailed descriptions.

### Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| TruthfulQA (MC validation) | Hugging Face (`truthful_qa`) | 817 rows | Truthfulness MCQA | datasets/truthful_qa_multiple_choice_validation/ | Targeted behavior evaluation |
| IFEval (train) | Hugging Face (`google/IFEval`) | 541 rows | Instruction following | datasets/google_ifeval_train/ | Multi-constraint composition |
| HellaSwag (validation) | Hugging Face (`Rowan/hellaswag`) | 10,042 rows | Commonsense MC completion | datasets/rowan_hellaswag_validation/ | Off-target capability retention |

See `datasets/README.md` for download/loading instructions.

### Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| function_vectors | https://github.com/ericwtodd/function_vectors | Official FV implementation | code/function_vectors/ | Best direct baseline for this topic |
| llm-steer-instruct | https://github.com/microsoft/llm-steer-instruct | Instruction steering + composition | code/llm-steer-instruct/ | Strong composition evaluation protocol |
| activation_steering | https://github.com/cma1114/activation_steering | Experimental steering analyses | code/activation_steering/ | Useful practical heuristics |
| pyvene | https://github.com/stanfordnlp/pyvene | Generic intervention library | code/pyvene/ | Reusable intervention API |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability toolkit | code/TransformerLens/ | Best for residual-stream probing |

See `code/README.md` for key files and commit hashes.

### Resource Gathering Notes

#### Search Strategy
- Attempted local `paper-finder` script first (`find_papers.py` in fast/diligent modes).
- Service/script did not return results in this environment within timeout, so manual arXiv-guided search was used.
- Prioritized papers on: function vectors, activation steering, compositional steering, and representation geometry.
- Downloaded full PDFs and chunked them with `pdf_chunker.py` for page-preserving review.

#### Selection Criteria
- Direct relevance to compositionality of residual-stream directions.
- Coverage of both foundational methods and recent alternatives (2023-2026).
- Preference for resources with usable public code and reproducible pipelines.

#### Challenges Encountered
- `uv add` failed because this repo is not packaged for editable install; used `uv pip install` fallback inside fresh `.venv`.
- `find_papers.py` timed out; switched to manual discovery.

#### Gaps and Workarounds
- Some newer papers do not expose ready-to-run repos in the same format as older steering papers.
- Workaround: include robust tool libraries (`TransformerLens`, `pyvene`) to reimplement methods consistently.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: IFEval + TruthfulQA; HellaSwag as off-target regression check.
2. **Baseline methods**: ActAdd, CAA, static additive composition, dynamic composition.
3. **Evaluation metrics**: target adherence, composition gain, interference index, off-target capability delta.
4. **Code to adapt/reuse**: `function_vectors` for extraction baseline, `llm-steer-instruct` for composition protocol, `TransformerLens` for controlled ablations and subspace tests.

## Research Execution Update (2026-04-05)

A full automated research run was executed in this workspace using the pre-gathered resources.

### What Was Executed
- Implemented end-to-end pipeline: `src/run_research.py`
- Environment: local `.venv` (isolated), dependencies frozen in `requirements.txt`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Data used:
  - `code/llm-steer-instruct/data/ifeval_wo_instructions.jsonl`
  - `datasets/truthful_qa_multiple_choice_validation`
  - `datasets/rowan_hellaswag_validation`

### Outputs Produced
- Core metrics and tables in `results/`
  - `summary.json`
  - `direction_metadata.csv`
  - `representation_metrics.csv`
  - `pairwise_compositionality.csv`
  - `transfer_locality.csv`
  - `behavioral_results.csv`
- Figures in `figures/`
  - `pairwise_compositionality_heatmap.png`
  - `related_vs_unrelated_boxplot.png`
  - `transfer_locality_barplot.png`
  - `behavioral_compositionality.png`
- Documentation
  - `planning.md`
  - `REPORT.md`
  - updated `README.md`

### Validation
- Reproducibility rerun completed: key summary metrics matched exactly between
  `results/summary_run1.json` and `results/summary_run2.json`.
