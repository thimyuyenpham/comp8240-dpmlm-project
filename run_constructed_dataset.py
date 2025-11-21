from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import re
import random

import numpy as np
import pandas as pd
from datasets import load_dataset
from faker import Faker
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
class AmazonConfig:
    n_examples: int = 200
    seed: int = 42


def build_amazon_with_pii(cfg: AmazonConfig) -> pd.DataFrame:
    """Create the Amazon + synthetic-PII dataset used in the report."""
    ds = load_dataset("amazon_polarity", split="train")
    ds = ds.shuffle(seed=cfg.seed).select(range(cfg.n_examples))

    label_map = {0: "negative", 1: "positive"}

    df = pd.DataFrame(
        {
            "title": ds["title"],
            "text_orig": ds["content"],
            "label": [label_map[int(y)] for y in ds["label"]],
        }
    )

    fake = Faker()
    random.seed(cfg.seed)

    def inject_pii(text: str):
        name = fake.name()
        phone = fake.phone_number()
        city = fake.city()
        template = f"My name is {name}, I live in {city}, and my phone number is {phone}."
        injected = f"{template} {text}"
        return injected, name, phone, city

    injected_rows = [inject_pii(t) for t in df["text_orig"]]

    df["text_with_pii"] = [r[0] for r in injected_rows]
    df["pii_name"] = [r[1] for r in injected_rows]
    df["pii_phone"] = [r[2] for r in injected_rows]
    df["pii_city"] = [r[3] for r in injected_rows]

    out_path = DATA_DIR / "amazon_with_injected_pii.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved Amazon + synthetic PII dataset to {out_path}")

    return df


PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-\s.]*)?(?:\(?\d{2,4}\)?[-\s.]*){2,4}\d{2,4}"
)


def contains_phone(text: str) -> bool:
    return bool(PHONE_RE.search(text or ""))


def contains_substring(text: str, sub: str) -> bool:
    t = (text or "").lower()
    s = (sub or "").lower()
    return bool(s) and (s in t)


def count_detected_pii(text: str, name: str, phone: str, city: str) -> int:
    """Return how many of the injected PII fields are still detectable."""
    n = 0
    if phone and contains_phone(text):
        n += 1
    if name and contains_substring(text, name):
        n += 1
    if city and contains_substring(text, city):
        n += 1
    return n


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


def run_amazon_experiments(
    epsilons: Sequence[float],
    cfg: AmazonConfig | None = None,
) -> None:
    if cfg is None:
        cfg = AmazonConfig()

    csv_path = DATA_DIR / "amazon_with_injected_pii.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded existing Amazon + PII dataset from {csv_path}")
    else:
        df = build_amazon_with_pii(cfg)

    dp = load_dpmlm(DPMLMConfig())

    metrics_rows = []
    all_rows = []

    df["token_len_before"] = df["text_with_pii"].str.split().str.len()

    for eps in epsilons:
        print(f"Running DP-MLM on Amazon data with ε={eps} ...")

        priv_texts = rewrite_many(
            dp,
            df["text_with_pii"].tolist(),
            epsilon=eps,
            max_tokens=128,
        )

        token_len_after = [len((t or "").split()) for t in priv_texts]
        len_ratios = [
            length_ratio(o, p)
            for o, p in zip(df["text_with_pii"], priv_texts)
        ]
        token_changes = [
            token_change_fraction(o, p)
            for o, p in zip(df["text_with_pii"], priv_texts)
        ]
        sims = cosine_similarities(df["text_with_pii"].tolist(), priv_texts)

        before = [
            count_detected_pii(text, name, phone, city)
            for text, name, phone, city in zip(
                df["text_with_pii"],
                df["pii_name"],
                df["pii_phone"],
                df["pii_city"],
            )
        ]
        after = [
            count_detected_pii(text, name, phone, city)
            for text, name, phone, city in zip(
                priv_texts,
                df["pii_name"],
                df["pii_phone"],
                df["pii_city"],
            )
        ]

        pii_removal_recall = []
        for b, a in zip(before, after):
            if b == 0:
                pii_removal_recall.append(np.nan)
            else:
                pii_removal_recall.append((b - a) / b)

        s_len = pd.Series(len_ratios)
        s_chg = pd.Series(token_changes)
        s_sim = pd.Series(sims)
        s_rec = pd.Series(pii_removal_recall, dtype="float")

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
                "mean_pii_removal_recall": float(s_rec.mean(skipna=True)),
                "std_pii_removal_recall": float(s_rec.std(skipna=True)),
                "prop_all_pii_removed": float((s_rec == 1.0).mean(skipna=True)),
            }
        )

        df_eps = df.copy()
        df_eps["epsilon"] = eps
        df_eps["text_dp"] = priv_texts
        df_eps["token_len_after"] = token_len_after
        df_eps["len_ratio"] = len_ratios
        df_eps["token_change"] = token_changes
        df_eps["cosine_sim"] = sims
        df_eps["pii_detected_before"] = before
        df_eps["pii_detected_after"] = after
        df_eps["pii_removal_recall"] = pii_removal_recall

        all_rows.append(df_eps)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "amazon_pii_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    all_df = pd.concat(all_rows, ignore_index=True)
    per_example_path = RESULTS_DIR / "amazon_pii_with_dp.csv"
    all_df.to_csv(per_example_path, index=False)

    print(f"Saved Amazon summary metrics to {metrics_path}")
    print(f"Saved Amazon per-example results to {per_example_path}")


if __name__ == "__main__":
    EPSILONS = [10.0, 50.0, 250.0]
    cfg = AmazonConfig(n_examples=5, seed=42)
    run_amazon_experiments(EPSILONS, cfg)
