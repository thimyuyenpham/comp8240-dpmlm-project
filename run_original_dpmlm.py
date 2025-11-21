from __future__ import annotations

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


def prepare_trustpilot_subset(n_examples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Load a small, reproducible subset of Trustpilot reviews."""
    ds = load_dataset("Kerassy/trustpilot-reviews-123k", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_examples))
    df = pd.DataFrame({"review": ds["review"], "stars": ds["stars"]})
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
        sim = cosine_similarity(v1, v2)[0, 0]
        sims.append(float(sim))
    return sims


def run_trustpilot_experiment(
    epsilons: Sequence[float],
    n_examples: int = 200,
    seed: int = 42,
) -> None:
    """Rewrite Trustpilot reviews for several epsilons and save metrics."""
    df = prepare_trustpilot_subset(n_examples=n_examples, seed=seed)

    # Save the CLEAN subset to data/
    subset_path = DATA_DIR / "trustpilot_subset.csv"
    df.to_csv(subset_path, index=False)
    print(f"Saved clean Trustpilot subset to {subset_path}")

    dp = load_dpmlm(DPMLMConfig())

    metrics_rows = []

    for eps in epsilons:
        print(f"Running DP-MLM on Trustpilot with ε={eps} ...")

        priv_texts = rewrite_many(
            dp,
            df["review"].tolist(),
            epsilon=eps,
            max_tokens=128,
        )
        df[f"text_dp_eps_{int(eps)}"] = priv_texts

        len_ratios = [
            length_ratio(o, p) for o, p in zip(df["review"], priv_texts)
        ]
        token_changes = [
            token_change_fraction(o, p)
            for o, p in zip(df["review"], priv_texts)
        ]
        sims = cosine_similarities(df["review"].tolist(), priv_texts)

        df[f"len_ratio_eps_{int(eps)}"] = len_ratios
        df[f"token_change_eps_{int(eps)}"] = token_changes
        df[f"cosine_sim_eps_{int(eps)}"] = sims

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

    # Save the DP-augmented dataframe (original + rewrites + per-example metrics) to results/
    dp_per_example_path = RESULTS_DIR / "trustpilot_with_dp.csv"
    df.to_csv(dp_per_example_path, index=False)
    print(f"Saved Trustpilot per-example DP results to {dp_per_example_path}")

    # Save the summary metrics to results/ 
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "trustpilot_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved Trustpilot summary metrics to {metrics_path}")


if __name__ == "__main__":
    EPSILONS = [10.0, 50.0, 250.0]
    run_trustpilot_experiment(EPSILONS, n_examples=5)
