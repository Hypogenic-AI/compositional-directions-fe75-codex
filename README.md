# Which Linear Directions Are Compositional?

This project tests whether residual-stream directions in a transformer LLM compose reliably under vector addition, or only approximately in constrained regions.

## Key Findings
- Representation-space composition is **partial**: mean cosine between observed and additive-predicted displacements is `0.518`.
- Composition quality is uneven: `12/85` instruction pairs have mean cosine `<= 0.3`, including `2` negative pairs.
- Related direction families were only slightly higher than unrelated (`0.571` vs `0.509`, `p=0.664`, not significant).
- In the evaluated steering subset, additive composition did **not** beat best-single steering (composition gain `0.000`).

## Reproduce
```bash
# from workspace root
source .venv/bin/activate
python src/run_research.py
```

Outputs are written to:
- `results/` (CSV/JSON metrics)
- `figures/` (plots)
- `logs/` (run logs)

## Project Structure
- `planning.md`: research plan and hypothesis decomposition
- `src/run_research.py`: full experiment pipeline
- `results/summary.json`: main summary metrics
- `REPORT.md`: full analysis and discussion
- `requirements.txt`: environment package snapshot

## Notes
- Model used: `Qwen/Qwen2.5-0.5B-Instruct`
- Hardware during execution: 2x RTX 3090 (24GB)
- Full run time: ~5 minutes per run in this environment
