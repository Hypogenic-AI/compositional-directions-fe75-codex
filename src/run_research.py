import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from scipy.stats import mannwhitneyu, wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        means.append(float(np.mean(values[idx])))
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = (a.norm() * b.norm()).item() + 1e-8
    return float(torch.dot(a, b).item() / denom)


def normed(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def parse_ifeval_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_constraint_lookup() -> Dict[str, str]:
    return {
        "punctuation:no_comma": " Do not use commas.",
        "detectable_format:title": " Include a title wrapped in double angle brackets like <<title>>.",
        "combination:repeat_prompt": " First repeat the request exactly, then answer.",
        "detectable_content:number_placeholders": " Include at least 3 placeholders in square brackets like [name].",
        "detectable_format:number_highlighted_sections": " Highlight at least 2 sections using markdown italics like *section*.",
    }


def instruction_family(instr: str) -> str:
    return instr.split(":", 1)[0] if ":" in instr else instr


@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    seed: int = 42
    max_len: int = 256
    n_single_per_id: int = 16
    n_multi_eval: int = 220
    min_single_support: int = 10
    transfer_n: int = 120
    behavior_n_per_pair: int = 8
    max_new_tokens: int = 72


class ResidualDirectionStudy:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

        # Qwen/GPT-like models expose decoder blocks at model.layers.
        self.blocks = self.model.model.layers
        self.n_layers = len(self.blocks)
        self.layer = self.n_layers // 2

    @torch.no_grad()
    def last_hidden(self, text: str) -> torch.Tensor:
        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_len,
        )
        toks = {k: v.to(self.model.device) for k, v in toks.items()}
        out = self.model(**toks, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.layer + 1][0, -1, :].detach().float().cpu()
        return h

    @torch.no_grad()
    def generate(self, prompt: str, steer_vec: torch.Tensor | None, coeff: float = 3.0) -> str:
        toks = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_len)
        toks = {k: v.to(self.model.device) for k, v in toks.items()}

        handle = None
        if steer_vec is not None:
            vec = (coeff * normed(steer_vec)).to(self.model.device)

            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    h = output[0] + vec.view(1, 1, -1).to(device=output[0].device, dtype=output[0].dtype)
                    return (h,) + output[1:]
                return output + vec.view(1, 1, -1).to(device=output.device, dtype=output.dtype)

            handle = self.blocks[self.layer].register_forward_hook(hook)

        gen = self.model.generate(
            **toks,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if handle is not None:
            handle.remove()

        out = self.tokenizer.decode(gen[0][toks["input_ids"].shape[1]:], skip_special_tokens=True)
        return out.strip()


def score_instruction(instr: str, text: str, base_prompt: str) -> int:
    t = text.strip()
    if instr == "punctuation:no_comma":
        return int("," not in t)
    if instr == "detectable_format:title":
        return int("<<" in t and ">>" in t)
    if instr == "combination:repeat_prompt":
        p = re.sub(r"\s+", " ", base_prompt.strip())
        g = re.sub(r"\s+", " ", t)
        return int(g.startswith(p[: max(30, min(len(p), 80))]))
    if instr == "detectable_content:number_placeholders":
        return int(t.count("[") >= 2 and t.count("]") >= 2)
    if instr == "detectable_format:number_highlighted_sections":
        return int(t.count("*") >= 4)
    if instr == "change_case:english_lowercase":
        letters = [c for c in t if c.isalpha()]
        return int(len(letters) > 0 and all(c.islower() for c in letters))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = RunConfig(model_name=args.model, seed=args.seed)
    set_seed(cfg.seed)

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    start_time = time.time()

    rows = parse_ifeval_jsonl("code/llm-steer-instruct/data/ifeval_wo_instructions.jsonl")

    single_by_id = defaultdict(list)
    multi_rows = []
    for r in rows:
        uniq = sorted(set(r["instruction_id_list"]))
        if len(uniq) == 1:
            single_by_id[uniq[0]].append(r)
        elif len(uniq) >= 2:
            multi_rows.append(r)

    support = {k: len(v) for k, v in single_by_id.items()}
    selected_ids = sorted([k for k, v in support.items() if v >= cfg.min_single_support])

    study = ResidualDirectionStudy(cfg)

    directions: Dict[str, torch.Tensor] = {}
    dir_meta = {}

    # Direction extraction + split-half stability.
    for instr in selected_ids:
        examples = single_by_id[instr][: cfg.n_single_per_id]
        diffs = []
        for ex in examples:
            h_full = study.last_hidden(ex["original_prompt"])
            h_base = study.last_hidden(ex["model_output"])
            diffs.append(h_full - h_base)
        if len(diffs) < 4:
            continue
        stack = torch.stack(diffs)
        d = stack.mean(dim=0)
        directions[instr] = d

        idx = np.arange(len(diffs))
        split_cos = []
        for _ in range(60):
            np.random.shuffle(idx)
            a = idx[: len(idx) // 2]
            b = idx[len(idx) // 2 :]
            if len(a) == 0 or len(b) == 0:
                continue
            da = stack[a].mean(dim=0)
            db = stack[b].mean(dim=0)
            split_cos.append(cos(da, db))

        dir_meta[instr] = {
            "n_used": len(diffs),
            "norm": float(d.norm().item()),
            "split_half_cos_mean": float(np.mean(split_cos)) if split_cos else float("nan"),
        }

    # Representation compositionality.
    rep_rows = []
    pair_rows = []
    random.shuffle(multi_rows)
    eval_multi = multi_rows[: cfg.n_multi_eval]

    for ex in eval_multi:
        ids = [i for i in sorted(set(ex["instruction_id_list"])) if i in directions]
        if len(ids) < 2:
            continue
        h_full = study.last_hidden(ex["original_prompt"])
        h_base = study.last_hidden(ex["model_output"])
        obs = h_full - h_base
        pred = sum(directions[i] for i in ids)
        c = cos(obs, pred)
        rel_norm_err = float(abs(obs.norm().item() - pred.norm().item()) / (obs.norm().item() + 1e-8))
        r2_like = float(1.0 - ((obs - pred).pow(2).sum().item() / (obs.pow(2).sum().item() + 1e-8)))
        rep_rows.append(
            {
                "key": ex["key"],
                "n_instr": len(ids),
                "instr": "|".join(ids),
                "cosine": c,
                "rel_norm_err": rel_norm_err,
                "r2_like": r2_like,
            }
        )

        if len(ids) == 2:
            a, b = ids
            pair_rows.append(
                {
                    "pair": f"{a} || {b}",
                    "a": a,
                    "b": b,
                    "cosine": c,
                    "rel_norm_err": rel_norm_err,
                    "r2_like": r2_like,
                    "related": int(instruction_family(a) == instruction_family(b)),
                }
            )

    rep_df = pd.DataFrame(rep_rows)
    pair_df = pd.DataFrame(pair_rows)

    # Transfer locality checks via synthetic instruction text on OOD prompts.
    constraint_text = build_constraint_lookup()
    usable_transfer_ids = [i for i in directions if i in constraint_text]
    transfer_pairs = list(combinations(usable_transfer_ids, 2))

    tqa = load_from_disk("datasets/truthful_qa_multiple_choice_validation")
    hs = load_from_disk("datasets/rowan_hellaswag_validation")
    tqa_prompts = [f"Question: {x['question']} Answer:" for x in tqa.select(range(min(cfg.transfer_n, len(tqa))))]
    hs_prompts = [x["ctx"] for x in hs.select(range(min(cfg.transfer_n, len(hs))))]

    transfer_rows = []
    for ds_name, prompts in [("truthfulqa", tqa_prompts), ("hellaswag", hs_prompts)]:
        for a, b in transfer_pairs:
            cvals = []
            eval_n = min(36, len(prompts))
            for p in prompts[:eval_n]:
                h0 = study.last_hidden(p)
                ha = study.last_hidden(p + constraint_text[a])
                hb = study.last_hidden(p + constraint_text[b])
                hab = study.last_hidden(p + constraint_text[a] + constraint_text[b])
                obs = hab - h0
                pred = (ha - h0) + (hb - h0)
                cvals.append(cos(obs, pred))
            transfer_rows.append(
                {
                    "dataset": ds_name,
                    "pair": f"{a} || {b}",
                    "a": a,
                    "b": b,
                    "cosine_mean": float(np.mean(cvals)),
                    "cosine_std": float(np.std(cvals)),
                }
            )

    transfer_df = pd.DataFrame(transfer_rows)

    # Behavioral steering on exactly-two-instruction examples with checkable constraints.
    behavior_ids = {
        "punctuation:no_comma",
        "detectable_format:title",
        "combination:repeat_prompt",
        "detectable_content:number_placeholders",
        "detectable_format:number_highlighted_sections",
        "change_case:english_lowercase",
    }

    pair_to_examples = defaultdict(list)
    for ex in rows:
        ids = [i for i in sorted(set(ex["instruction_id_list"])) if i in behavior_ids and i in directions]
        if len(ids) == 2:
            pair_to_examples[(ids[0], ids[1])].append(ex)

    candidate_pairs = sorted(pair_to_examples.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected_pairs = [p for p, exs in candidate_pairs if len(exs) >= 4][:4]

    beh_rows = []
    for a, b in selected_pairs:
        examples = pair_to_examples[(a, b)][: cfg.behavior_n_per_pair]
        for ex in examples:
            prompt = ex["model_output"]
            conds = {
                "none": None,
                "single_a": directions[a],
                "single_b": directions[b],
                "sum": directions[a] + directions[b],
            }
            for cname, vec in conds.items():
                txt = study.generate(prompt, steer_vec=vec, coeff=3.0)
                sa = score_instruction(a, txt, prompt)
                sb = score_instruction(b, txt, prompt)
                beh_rows.append(
                    {
                        "key": ex["key"],
                        "pair": f"{a} || {b}",
                        "a": a,
                        "b": b,
                        "condition": cname,
                        "score_a": sa,
                        "score_b": sb,
                        "joint": int(sa and sb),
                        "text": txt,
                    }
                )

    beh_df = pd.DataFrame(beh_rows)

    # Summaries and stats.
    summary = {
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "device": study.device,
        "n_layers": study.n_layers,
        "steer_layer": study.layer,
        "n_selected_directions": len(directions),
        "selected_direction_ids": sorted(directions.keys()),
        "direction_meta": dir_meta,
        "representation_mean_cosine": float(rep_df["cosine"].mean()) if len(rep_df) else float("nan"),
        "representation_mean_r2_like": float(rep_df["r2_like"].mean()) if len(rep_df) else float("nan"),
        "representation_mean_rel_norm_err": float(rep_df["rel_norm_err"].mean()) if len(rep_df) else float("nan"),
        "run_seconds": float(time.time() - start_time),
        "gpu_available": bool(torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()),
    }

    if len(pair_df):
        related = pair_df[pair_df["related"] == 1]["cosine"].to_numpy()
        unrelated = pair_df[pair_df["related"] == 0]["cosine"].to_numpy()
        if len(related) and len(unrelated):
            try:
                mw = mannwhitneyu(related, unrelated, alternative="two-sided")
                summary["related_vs_unrelated_mwu_p"] = float(mw.pvalue)
                summary["related_cosine_mean"] = float(np.mean(related))
                summary["unrelated_cosine_mean"] = float(np.mean(unrelated))
            except Exception:
                pass
        lo, hi = bootstrap_ci(pair_df["cosine"].to_numpy())
        summary["pair_cosine_bootstrap_ci95"] = [lo, hi]

    if len(beh_df):
        piv = beh_df.groupby(["pair", "condition"], as_index=False)["joint"].mean()
        # Compare additive vs best single per pair.
        gains = []
        for pair_name in piv["pair"].unique():
            sub = piv[piv["pair"] == pair_name].set_index("condition")
            if {"sum", "single_a", "single_b"}.issubset(set(sub.index)):
                gains.append(float(sub.loc["sum", "joint"] - max(sub.loc["single_a", "joint"], sub.loc["single_b", "joint"])))
        if gains:
            summary["behavior_mean_composition_gain_over_best_single"] = float(np.mean(gains))
            lo, hi = bootstrap_ci(np.array(gains))
            summary["behavior_composition_gain_ci95"] = [lo, hi]

    # Save tabular outputs.
    pd.DataFrame(
        [{"instruction": k, **v} for k, v in sorted(dir_meta.items())]
    ).to_csv("results/direction_metadata.csv", index=False)
    rep_df.to_csv("results/representation_metrics.csv", index=False)
    pair_df.to_csv("results/pairwise_compositionality.csv", index=False)
    transfer_df.to_csv("results/transfer_locality.csv", index=False)
    beh_df.to_csv("results/behavioral_results.csv", index=False)
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save vectors in a compact json format.
    dir_payload = {k: directions[k].tolist() for k in sorted(directions.keys())}
    with open("results/directions.json", "w", encoding="utf-8") as f:
        json.dump(dir_payload, f)

    # Figures.
    sns.set_theme(style="whitegrid")

    if len(pair_df):
        pivot = pair_df.groupby(["a", "b"], as_index=False)["cosine"].mean()
        mat = pivot.pivot(index="a", columns="b", values="cosine")
        plt.figure(figsize=(10, 7))
        sns.heatmap(mat, cmap="coolwarm", center=0.0)
        plt.title("Pairwise Compositionality (Mean Cosine)")
        plt.tight_layout()
        plt.savefig("figures/pairwise_compositionality_heatmap.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=pair_df, x="related", y="cosine")
        plt.xticks([0, 1], ["Unrelated", "Related"])
        plt.ylabel("Cosine(obs, additive prediction)")
        plt.title("Related vs Unrelated Direction Pairs")
        plt.tight_layout()
        plt.savefig("figures/related_vs_unrelated_boxplot.png", dpi=180)
        plt.close()

    if len(transfer_df):
        plt.figure(figsize=(11, 4))
        sns.barplot(data=transfer_df, x="pair", y="cosine_mean", hue="dataset")
        plt.xticks(rotation=85, ha="right")
        plt.ylabel("Mean transfer compositionality cosine")
        plt.title("Transfer Locality: Additivity on OOD Contexts")
        plt.tight_layout()
        plt.savefig("figures/transfer_locality_barplot.png", dpi=180)
        plt.close()

    if len(beh_df):
        beh_plot = beh_df.groupby(["pair", "condition"], as_index=False)["joint"].mean()
        plt.figure(figsize=(10, 4))
        sns.barplot(data=beh_plot, x="pair", y="joint", hue="condition")
        plt.ylabel("Joint constraint satisfaction")
        plt.xticks(rotation=70, ha="right")
        plt.title("Behavioral Compositionality Under Steering")
        plt.tight_layout()
        plt.savefig("figures/behavioral_compositionality.png", dpi=180)
        plt.close()

    # Environment + run metadata.
    env = {
        "python": os.popen("python -V").read().strip(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_name_1": torch.cuda.get_device_name(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
    }
    with open("results/environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
