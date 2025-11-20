from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dpmlm_utils import (
    DPMLMConfig,
    load_dpmlm,
    rewrite_many,
    length_ratio,
    token_change_fraction,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AmazonPlainConfig:
    n_examples: int = 3   # how many Amazon reviews to rewrite
    seed: int = 42


def prepare_amazon_plain(cfg: AmazonPlainConfig) -> pd.DataFrame:
    """Load a small subset of Amazon Polarity reviews."""
    ds = load_dataset("amazon_polarity", split="train")
    ds = ds.shuffle(seed=cfg.seed).select(range(cfg.n_examples))

    label_map = {0: "negative", 1: "positive"}

    df = pd.DataFrame(
        {
            "title": ds["title"],
            "text": ds["content"],
            "label": [label_map[int(y)] for y in ds["label"]],
        }
    )

    out_path = DATA_DIR / "amazon_plain_subset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved Amazon plain subset to {out_path}")
    return df


def cosine_similarities(orig: list[str], priv: list[str]) -> list[float]:
    """TF-IDF cosine similarity between original and privatized texts."""
    all_texts = orig + priv
    vect = TfidfVectorizer().fit_transform(all_texts)

    n = len(orig)
    sims: list[float] = []
    for i in range(n):
        v1 = vect[i]
        v2 = vect[i + n]
        sims.append(float(cosine_similarity(v1, v2)[0, 0]))
    return sims


def run_amazon_plain_experiments(
    epsilons: Sequence[float],
    cfg: AmazonPlainConfig | None = None,
) -> None:
    """Rewrite plain Amazon reviews with DP-MLM and save metrics + per-example CSVs."""
    if cfg is None:
        cfg = AmazonPlainConfig()

    csv_path = DATA_DIR / "amazon_plain_subset.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded Amazon plain subset from {csv_path}")
        # ensure we only use cfg.n_examples rows even if file is bigger
        if cfg.n_examples is not None and cfg.n_examples < len(df):
            df = df.sample(n=cfg.n_examples, random_state=cfg.seed).reset_index(drop=True)
    else:
        df = prepare_amazon_plain(cfg)

    dp = load_dpmlm(DPMLMConfig())

    metrics_rows = []
    all_rows = []

    for eps in epsilons:
        print(f"Running DP-MLM on Amazon plain with ε={eps} ...")

        priv_texts = rewrite_many(
            dp,
            df["text"].tolist(),
            epsilon=eps,
            max_tokens=128,
        )

        len_ratios = [
            length_ratio(o, p) for o, p in zip(df["text"], priv_texts)
        ]
        token_changes = [
            token_change_fraction(o, p)
            for o, p in zip(df["text"], priv_texts)
        ]
        sims = cosine_similarities(df["text"].tolist(), priv_texts)

        s_len = pd.Series(len_ratios)
        s_chg = pd.Series(token_changes)
        s_sim = pd.Series(sims)

        metrics_rows.append(
            {
                "epsilon": eps,
                "n_examples": len(df),
                "mean_len_ratio": float(s_len.mean()),
                "std_len_ratio": float(s_len.std()),
                "mean_token_change": float(s_chg.mean()),
                "std_token_change": float(s_chg.std()),
                "mean_cosine_sim": float(s_sim.mean()),
                "std_cosine_sim": float(s_sim.std()),
            }
        )

        df_eps = df.copy()
        df_eps["epsilon"] = eps
        df_eps["text_dp"] = priv_texts
        df_eps["len_ratio"] = len_ratios
        df_eps["token_change"] = token_changes
        df_eps["cosine_sim"] = sims
        all_rows.append(df_eps)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "amazon_plain_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    per_example_df = pd.concat(all_rows, ignore_index=True)
    per_example_path = RESULTS_DIR / "amazon_plain_per_example.csv"
    per_example_df.to_csv(per_example_path, index=False)

    print(f"Saved Amazon plain metrics to {metrics_path}")
    print(f"Saved Amazon plain per-example results to {per_example_path}")


if __name__ == "__main__":
    EPSILONS = [10.0, 50.0, 250.0]
    cfg = AmazonPlainConfig(n_examples=5, seed=42)
    run_amazon_plain_experiments(EPSILONS, cfg)
