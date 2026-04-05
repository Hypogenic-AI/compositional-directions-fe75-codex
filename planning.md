## Research Question
Which residual-stream linear directions in a transformer are truly compositional under vector addition, and which are only locally/approximately linear?

## Motivation & Novelty Assessment

### Why This Research Matters
Activation steering and mechanistic-interpretability claims often assume that discovered residual directions can be safely added and composed. If that assumption fails outside narrow settings, safety controls, controllability, and interpretability claims become brittle. Mapping where compositionality holds versus fails is therefore directly useful for reliable steering methods.

### Gap in Existing Work
Prior work (ActAdd, CAA, Function Vectors, instruction steering) shows average steering gains but gives limited systematic maps of compositional vs non-compositional directions across instruction families, including instance-level reliability and failure regions. Recent vector-field work suggests local rather than global linearity, but direct per-direction compositional diagnostics remain underdeveloped.

### Our Novel Contribution
We build a direction-level compositionality map from real residual activations in a modern open model, using both representation-space and behavior-space tests. We explicitly separate:
- additive compatibility of related directions,
- interference among unrelated directions,
- transfer of direction geometry across datasets/contexts.

### Experiment Justification
- Experiment 1: Representation-space compositionality on IFEval instruction vectors.
  - Why needed: directly tests linear-subspace assumption where observed displacement can be compared to sum of single-instruction directions.
- Experiment 2: Behavioral compositionality via activation steering on automatically checkable constraints.
  - Why needed: geometry may look linear but fail causally during generation; this verifies practical composability.
- Experiment 3: Robustness and transfer checks on TruthfulQA and HellaSwag contexts.
  - Why needed: tests whether compositional directions remain coherent outside the training/discovery manifold.

## Background and Motivation
The core skepticism is that not all residual directions form a coherent linear space with stable additive composition. The project tests this skepticism directly by comparing additive predictions against observed combined effects and by identifying which direction families (e.g., related formatting constraints) are more additive than others.

## Hypothesis Decomposition
- H1 (partial compositionality): Some semantically related instruction directions are compositional (high alignment between observed combined displacement and vector sums).
- H2 (non-universality): Many directions are not compositional globally (low alignment, norm mismatch, high variance across contexts).
- H3 (behavioral mismatch): Even when representational additivity is moderate, behavioral constraint satisfaction under summed steering degrades for some direction pairs.
- H4 (locality): Compositionality scores drop when evaluated on out-of-family contexts (TruthfulQA/HellaSwag), indicating local linearity.

Independent variables:
- Direction type (instruction family)
- Pair relatedness (related/unrelated)
- Steering condition (none, single A, single B, A+B)
- Steering coefficient and layer

Dependent variables:
- Representation compositionality: cosine(obs, sum), relative norm error, R^2 fit
- Behavioral compositionality: dual-constraint satisfaction, gain over best single, interference score
- Robustness: score shift under context transfer

Alternative explanations considered:
- Effects caused by layer/scale mismatch
- Apparent nonlinearity from poor direction estimation due to low sample count
- Prompt-length and lexical confounds

## Proposed Methodology

### Approach
Use a real pretrained transformer with hidden-state access (Qwen2.5-1.5B-Instruct) and extract residual directions from real datasets (IFEval + provided simplification file). Then test compositionality in two complementary ways: (1) representational additivity and (2) causal steering behavior on constraints with automatic checkers.

### Experimental Steps
1. Data preparation and filtering
- Parse IFEval and `ifeval_wo_instructions.jsonl` to form single-instruction and multi-instruction subsets.
- Keep instruction IDs with sufficient support for reliable estimation.

2. Direction extraction (baseline)
- For each instruction ID (single-instruction examples), compute direction as mean residual difference between constrained and de-constrained prompts.
- Evaluate estimation stability by bootstrap split-half cosine.

3. Representation compositionality test
- For multi-instruction examples, compute observed residual displacement.
- Predict displacement by summing constituent single directions.
- Compute compositionality metrics per pair/family and aggregate with confidence intervals.

4. Behavioral steering test
- Select auto-checkable instructions (e.g., lowercase, no_comma, quotation, bullet-list constraints) and compositional pairs with available data.
- Generate responses from de-constrained prompts under: none, A, B, A+B steering.
- Score constraints automatically and compute composition gain/interference.

5. Robustness transfer checks
- Project discovered directions onto TruthfulQA and HellaSwag prompt activations.
- Measure whether pairwise compositionality geometry degrades off-distribution.

6. Analysis and visualization
- Heatmap of pairwise compositionality.
- Related-vs-unrelated boxplots.
- Behavioral bar plots with error bars.

### Baselines
- No-steering baseline.
- Single-direction steering baseline.
- Best-single oracle (upper baseline for additive method comparison).
- Random direction control (same norm as real vectors).

### Evaluation Metrics
- Representation: cosine alignment, norm ratio error, explained variance (R^2), split-half reliability.
- Behavioral: instruction pass rate per constraint, joint pass rate, composition gain, interference index.
- Transfer: drop in compositionality metrics between in-domain (IFEval) and transfer contexts.

### Statistical Analysis Plan
- Alpha = 0.05.
- Paired comparisons with Wilcoxon signed-rank (robust for non-normal metric distributions).
- Effect size: rank-biserial correlation and Cliff’s delta where applicable.
- Bootstrap 95% confidence intervals (1,000 resamples).
- Benjamini-Hochberg correction for multiple pairwise tests.

## Expected Outcomes
- Support for hypothesis: related direction families show higher compositionality than unrelated ones; additive steering helps some pairs but not all; transfer to TruthfulQA/HellaSwag reduces compositionality.
- Refutation: consistently high compositionality across most directions and contexts.

## Timeline and Milestones
- M1 (setup + planning): 20-30 min
- M2 (implementation): 60-90 min
- M3 (experiments): 60-90 min
- M4 (analysis + report): 45-60 min
- Buffer/debug: 25%

## Potential Challenges
- Model generation speed for steering loops.
  - Mitigation: cap sample sizes; batch activation extraction; run on GPU.
- Sparse instruction combinations.
  - Mitigation: focus on top-support instruction IDs and documented exclusions.
- Hooking fragility across transformer architectures.
  - Mitigation: architecture-specific hook wrapper with assertions and fallback layer choices.

## Success Criteria
- Complete compositionality map over a non-trivial set of instruction directions (>=8 IDs).
- Behavioral evaluation on >=3 compositional pairs with automatic scoring.
- Statistical comparison supporting or rejecting each sub-hypothesis with CIs.
- Reproducible scripts and documented outputs in `results/`, `figures/`, and `REPORT.md`.
