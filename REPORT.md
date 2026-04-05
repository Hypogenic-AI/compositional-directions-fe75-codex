# Which Linear Directions Are Compositional?

## 1. Executive Summary
We tested whether residual-stream directions in a transformer LLM compose linearly, or only approximately in limited regions. Using Qwen2.5-0.5B-Instruct and instruction-derived direction vectors from IFEval, we found **partial compositionality**: additive predictions were moderately aligned on average (mean cosine 0.518), but many samples/pairs showed poor fit (mean R2-like -0.495, 12/85 pairs <= 0.3 cosine, 2/85 negative).

Practical implication: treating residual directions as a globally compositional vector space is too strong. Some directions compose well, but composition is uneven and behavior-level gains from naive vector summation were negligible in our steering subset.

## 2. Research Question & Motivation
### Hypothesis
Not all linear residual directions form coherent compositional subspaces; related directions may compose more naturally than unrelated ones.

### Why It Matters
Activation steering and interpretability often assume additive direction arithmetic. If that assumption is not robust, safety controls and controllability methods can fail outside narrow settings.

### Gap Addressed
Prior work reports average steering success but gives limited direction-level maps of where composition fails. We provide a systematic, per-direction compositionality map with representation-level and behavior-level checks.

## 3. Methodology
### 3.1 Experimental Setup
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Libraries: `torch 2.5.1+cu124`, `transformers 5.5.0`, `datasets 4.8.4`, `scipy 1.17.1`, `pandas 3.0.2`
- Hardware: 2x NVIDIA RTX 3090 (24GB each)
- Steering layer: middle layer (`12` of `24`)
- Seed: `42`
- Runtime per full run: ~299s

### 3.2 Data
- IFEval-derived paired prompts from `code/llm-steer-instruct/data/ifeval_wo_instructions.jsonl` (541 rows)
- TruthfulQA validation (`817`) for transfer checks
- HellaSwag validation (`10042`) for transfer checks

### 3.3 Direction Discovery
For each instruction ID with >=10 single-instruction examples:
1. Compute last-token residual vector on constrained prompt.
2. Compute last-token residual vector on de-constrained prompt (`model_output`).
3. Direction = mean difference over examples.

Extracted directions: 20 instruction IDs (see `results/direction_metadata.csv`).

### 3.4 Experiment 1: Representation Compositionality
For multi-instruction prompts, compare:
- observed displacement: `h(constrained) - h(base)`
- additive prediction: `sum(direction_i)` for involved instructions

Metrics:
- cosine alignment
- relative norm error
- R2-like reconstruction score

### 3.5 Experiment 2: Behavioral Compositionality (Steering)
For exactly-two-instruction prompts (checkable constraints), generate with:
- no steering
- steering by A
- steering by B
- steering by A+B

Scoring: automatic rule checks (e.g., no comma, title markers, repeated prompt prefix, placeholders).

### 3.6 Experiment 3: Transfer/Locality Check
On TruthfulQA and HellaSwag prompts, append synthetic instruction snippets and test whether combined perturbation is additive:
- `obs = h(x+i+j)-h(x)`
- `pred = (h(x+i)-h(x)) + (h(x+j)-h(x))`
- evaluate cosine(obs, pred)

### 3.7 Statistical Plan
- Related-vs-unrelated pair comparison: Mann-Whitney U
- Bootstrap CI (95%) for mean pairwise cosine and behavioral composition gain
- Reproducibility check: full rerun with same seed and metric comparison

## 4. Results
### 4.1 Main Quantitative Results
| Metric | Value |
|---|---:|
| Selected directions | 20 |
| Multi-instruction examples evaluated | 174 |
| Representation mean cosine | 0.518 |
| Representation mean R2-like | -0.495 |
| Representation mean relative norm error | 0.256 |
| Pairwise cosine 95% bootstrap CI | [0.478, 0.543] |
| Related-pair mean cosine | 0.571 |
| Unrelated-pair mean cosine | 0.509 |
| Related vs unrelated p-value | 0.664 |
| Behavioral composition gain over best single | 0.000 |

### 4.2 Pairwise Composition Map
- High-composition examples (mean cosine):
  - `detectable_format:title || keywords:letter_frequency` (0.767)
  - `change_case:capital_word_frequency || detectable_format:title` (0.766)
  - `change_case:english_capital || startend:end_checker` (0.761)
- Low-composition examples:
  - `length_constraints:number_paragraphs || startend:end_checker` (-0.176)
  - `detectable_format:number_bullet_lists || length_constraints:number_words` (-0.025)

### 4.3 Transfer Locality Results
Mean transfer additivity cosine:
- TruthfulQA: 0.868
- HellaSwag: 0.919

Interpretation: local appended-instruction perturbations remain highly additive in these contexts, even while full constrained-to-base displacements in IFEval are only moderately compositional.

### 4.4 Behavioral Steering Results
Joint constraint satisfaction rates:
- none: 0.000
- single A: 0.043
- single B: 0.043
- A+B: 0.043

Additive steering did **not** improve over best single in this evaluated subset.

### 4.5 Artifacts
- Tables: `results/*.csv`
- Summary metrics: `results/summary.json`
- Direction vectors: `results/directions.json`
- Figures:
  - `figures/pairwise_compositionality_heatmap.png`
  - `figures/related_vs_unrelated_boxplot.png`
  - `figures/transfer_locality_barplot.png`
  - `figures/behavioral_compositionality.png`

## 5. Analysis & Discussion
### What Supports the Hypothesis
- Compositionality is **not uniform**: pair quality varies widely (including negative pair cosines).
- Moderate average cosine with poor average reconstruction (negative mean R2-like) indicates additive direction sums are often directionally aligned but quantitatively miscalibrated.
- Behavioral additive steering did not outperform best-single steering, consistent with limited practical composability.

### What Does Not Strongly Support It
- Related vs unrelated direction families were not significantly different (p=0.664) in this sample.
- Transfer locality test showed high additivity for synthetic prompt perturbations, suggesting some local linear structure persists outside IFEval.

### Interpretation
A coherent global linear subspace is not supported. The data is more consistent with a mixed regime:
- local additive behavior for certain small perturbations,
- uneven compositionality for full instruction bundles,
- practical steering composition that is fragile or weak without better calibration.

## 6. Limitations
- Single model scale (0.5B) and one steering layer; larger models/layer sweeps may differ.
- Behavioral checkers are heuristic and only cover a subset of instruction types.
- Steering coefficient was fixed; no exhaustive coefficient tuning.
- Transfer test used synthetic instruction text appends (a local perturbation proxy), not full behavioral tasks.
- Cost tracking for API models not applicable here because this was a local-weight mechanistic setup.

## 7. Conclusions & Next Steps
Residual directions are partially compositional, but not reliably so across all direction pairs. The evidence supports the user’s skepticism: additive composition works for some direction families and fails for others, with non-trivial reconstruction error and weak behavior-level composition gains.

Recommended follow-up:
1. Run coefficient/layer sweeps per pair and estimate pair-specific optimal scaling.
2. Repeat on larger open models (e.g., 1.5B/7B) to test scale effects on compositionality.
3. Replace static-vector composition with context-dependent vector fields and compare directly.
4. Use stricter task-level evaluators (IFEval official scoring and longer-output constraints).

## Reproducibility Notes
- Environment: `.venv` created in workspace; dependencies in `requirements.txt`
- Main script: `src/run_research.py`
- Re-run command:
```bash
source .venv/bin/activate
python src/run_research.py
```
- Determinism check: a second full rerun produced identical key summary metrics (`results/summary_run1.json` vs `results/summary_run2.json`).

## References
- Todd et al., *Function Vectors in Large Language Models* (ICLR 2024)
- Turner et al., *Steering Language Models With Activation Engineering* (2023)
- Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition* (ACL 2024)
- Stolfo et al., *Improving Instruction-Following through Activation Steering* (ICLR 2025)
- Li et al., *Steering Vector Fields for Context-Aware Inference-Time Control in LLMs* (2026)
- Shai et al., *Transformers learn factored representations* (2026)
