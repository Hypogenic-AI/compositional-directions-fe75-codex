## Literature Review

### Research Area Overview
This review focuses on whether linear directions in transformer residual streams are truly compositional. Recent work shows that many behaviors can be steered by adding vectors to hidden states, but reliability varies by concept and context. The most relevant thread combines mechanistic interpretability (where in the network a feature lives), activation steering (whether a direction causally controls behavior), and representational geometry (whether features form additive, factorized, or entangled structures).

Search keywords used: `residual stream linear directions`, `activation steering`, `function vectors`, `compositional steering`, `linear representation hypothesis`, `factored representations`.

Note: the local paper-finder script timed out in this environment, so manual arXiv discovery and targeted selection were used.

### Key Papers

#### Paper 1: Function Vectors in Large Language Models
- **Authors**: Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, David Bau
- **Year**: 2023 (ICLR 2024)
- **Source**: arXiv / ICLR
- **Key Contribution**: Introduces function vectors (FVs) as compact causal representations of tasks in residual activations.
- **Methodology**: Causal mediation and head-level analysis; identify influential heads, sum their outputs into task vectors, inject at inference.
- **Datasets Used**: Task collections (e.g., antonyms/synonyms, translation, capitals, tense/plural transformations).
- **Results**: Strong steering effects in middle layers; partial semantic composition via vector addition for some task combinations.
- **Code Available**: Yes (`code/function_vectors`)
- **Relevance to Our Research**: Direct evidence that some vectors compose, but only partially and task-dependently.

#### Paper 2: Steering Llama 2 via Contrastive Activation Addition
- **Authors**: Nina Panickssery et al.
- **Year**: 2023 (ACL 2024)
- **Source**: arXiv / ACL
- **Key Contribution**: Contrastive Activation Addition (CAA) for behavioral steering from positive/negative prompt pairs.
- **Methodology**: Difference vectors in residual stream; add at generation-time across tokens.
- **Datasets Used**: Behavioral multiple-choice datasets and open-ended generation settings.
- **Results**: Effective control of targeted behaviors with limited capability degradation.
- **Code Available**: Indirect references in activation steering ecosystem.
- **Relevance to Our Research**: Strong baseline for testing linearity versus context dependence of directions.

#### Paper 3: Steering Language Models With Activation Engineering
- **Authors**: Alexander Matt Turner et al.
- **Year**: 2023
- **Source**: arXiv
- **Key Contribution**: ActAdd method for simple activation-space steering.
- **Methodology**: Build steering vectors from prompt-pair activation differences and inject during forward pass.
- **Datasets Used**: Sentiment, detoxification, topic/behavior control tasks.
- **Results**: Large behavioral shifts without retraining; robust enough for broad use.
- **Code Available**: Multiple public repos, including derivative tooling.
- **Relevance to Our Research**: Foundational linear-direction intervention method.

#### Paper 4: Improving Instruction-Following through Activation Steering
- **Authors**: Alessandro Stolfo et al.
- **Year**: 2024 (ICLR 2025)
- **Source**: arXiv / ICLR
- **Key Contribution**: Instruction-specific steering vectors, including multi-instruction composition.
- **Methodology**: Compute vectors from with/without-instruction pairs; tune layer and steering scale.
- **Datasets Used**: IFEval-style instruction benchmarks (format, length, keyword constraints).
- **Results**: Better adherence to constraints; composition works but quality-control tradeoffs remain.
- **Code Available**: Yes (`code/llm-steer-instruct`)
- **Relevance to Our Research**: Direct compositionality testbed for vector addition under multiple constraints.

#### Paper 5: HyperSteer
- **Authors**: Jiuding Sun et al.
- **Year**: 2025
- **Source**: arXiv
- **Key Contribution**: Hypernetwork-generated steering vectors at scale.
- **Methodology**: Condition a generator on steering prompts and model internals; produce vectors end-to-end.
- **Datasets Used**: Large prompt sets for steering and held-out steering tasks.
- **Results**: Better scaling behavior than static handcrafted vectors.
- **Code Available**: Not confirmed in this workspace.
- **Relevance to Our Research**: Baseline for “nonlinear/vector-generator” alternatives if linear compositionality fails.

#### Paper 6: Steering Vector Fields
- **Authors**: Jiaqian Li, Yanshu Li, Kuan-Hao Huang
- **Year**: 2026
- **Source**: arXiv
- **Key Contribution**: Replaces single static vectors with context-dependent vector fields.
- **Methodology**: Learn concept score; use local gradient as steering direction.
- **Datasets Used**: Multi-model steering tasks and long-form settings.
- **Results**: Improved reliability and multi-attribute control over static vectors.
- **Code Available**: Not confirmed in this workspace.
- **Relevance to Our Research**: Strong evidence that many directions are only locally linear.

#### Paper 7: Transformers Learn Factored Representations
- **Authors**: Adam Shai et al.
- **Year**: 2026
- **Source**: arXiv
- **Key Contribution**: Formalizes product-space vs factorized residual representation hypotheses.
- **Methodology**: Theory + synthetic latent-factor experiments; geometric tests for subspace structure.
- **Datasets Used**: Synthetic processes with known latent structure.
- **Results**: Transformers prefer factored/orthogonal representations when latent factors are near-conditionally-independent.
- **Code Available**: Not verified here.
- **Relevance to Our Research**: Provides explicit hypothesis testing framework for compositional subspaces.

#### Paper 8: Progress Measures for Grokking via Mechanistic Interpretability
- **Authors**: Neel Nanda et al.
- **Year**: 2023
- **Source**: arXiv / ICLR
- **Key Contribution**: Mechanistic progress metrics and phase decomposition of learned circuits.
- **Methodology**: Circuit reverse engineering + ablations + Fourier-space analyses.
- **Datasets Used**: Modular arithmetic synthetic tasks.
- **Results**: Emergent behavior can be tracked via continuous internal measures.
- **Code Available**: Tooling in mech-interp ecosystem (`TransformerLens` supports similar workflows).
- **Relevance to Our Research**: Useful methodology for tracking when compositional directions emerge during training/fine-tuning.

#### Paper 9: Dynamic Activation Composition
- **Authors**: Daniel Scalena, Gabriele Sarti, Malvina Nissim
- **Year**: 2024
- **Source**: arXiv
- **Key Contribution**: Dynamic weighting for multi-property steering.
- **Methodology**: Information-theoretic modulation of steering intensities over generation.
- **Datasets Used**: Multi-property steering evaluations.
- **Results**: Better simultaneous conditioning with reduced fluency loss.
- **Code Available**: Not verified in this workspace.
- **Relevance to Our Research**: Important for evaluating whether additive composition needs dynamic coefficients.

### Common Methodologies
- Prompt-pair vector extraction: Used in ActAdd/CAA/instruction-steering papers.
- Residual-stream injection: Most studies steer by adding vectors at selected layers/tokens.
- Layer search and coefficient sweeps: Standard for finding effective intervention points.
- Causal analysis / ablations: Used to separate correlation from causal control.

### Standard Baselines
- Prompt-only control (no activation intervention).
- Single-vector steering (one property at a time).
- Static additive multi-vector composition (`v_total = v1 + v2 + ...`).
- Finetuned instruction-following models for comparison.

### Evaluation Metrics
- Constraint satisfaction / instruction accuracy (IFEval-style).
- Target behavior shift metrics (MC accuracy deltas, attribute classifiers).
- Capability-retention metrics on off-target tasks.
- Generation quality proxies (perplexity or judge-based quality scoring).

### Datasets in the Literature
- IFEval and instruction-following subsets: used for controllable format/length/keyword constraints.
- Truthfulness datasets (TruthfulQA and variants): common for honesty/factuality steering.
- Synthetic latent-factor tasks: used to test representation factorization.
- Multi-property steering evaluation sets: used to test composition and interference.

### Gaps and Opportunities
- Gap 1: Many papers show average steering success, but limited per-instance reliability analyses.
- Gap 2: Global linear vectors often fail under context shift, long generation, or multi-attribute composition.
- Gap 3: Few works explicitly test whether discovered vectors define stable linear subspaces across prompts/models.
- Gap 4: Composition often tested for a small set of compatible attributes; incompatibility regimes are underexplored.

### Recommendations for Our Experiment
Based on gathered resources:
- **Recommended datasets**:
  - `google/IFEval` for controlled multi-constraint composition tests.
  - `truthful_qa` for truthfulness direction robustness and transfer.
  - `Rowan/hellaswag` for capability-retention checks.
- **Recommended baselines**:
  - ActAdd-style single-vector steering.
  - CAA with dataset-derived contrastive vectors.
  - Static additive composition vs dynamic weighting.
- **Recommended metrics**:
  - Target-constraint accuracy, composition gain, and interference score.
  - Off-target performance delta (e.g., HellaSwag change).
  - Stability across prompts/layers/scales (variance and worst-case performance).
- **Methodological considerations**:
  - Evaluate both average and worst-case behavior.
  - Report layer-locality and coefficient sensitivity.
  - Explicitly test linear-subspace assumptions by projecting and composing within/orthogonal to discovered bases.
