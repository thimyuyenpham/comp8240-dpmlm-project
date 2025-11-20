from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List
import sys
import math

# We assume DPMLM.py is in the same folder as this file
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import DPMLM  # type: ignore


@dataclass
class DPMLMConfig:
    """Simple configuration for creating a DP-MLM model."""
    model: str = "roberta-base"
    spacy_model: str = "en_core_web_md"
    alpha: float = 0.003


def load_dpmlm(config: DPMLMConfig | None = None) -> "DPMLM.DPMLM":
    """Create and return a DPMLM instance with default settings."""
    if config is None:
        config = DPMLMConfig()

    dp = DPMLM.DPMLM(
        MODEL=config.model,
        SPACY=config.spacy_model,
        alpha=config.alpha,
    )
    dp.load_transformers()
    return dp


def _truncate(text: str, max_tokens: int | None) -> str:
    """Truncate very long inputs for runtime safety."""
    if max_tokens is None:
        return text
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def rewrite_many(
    dp: "DPMLM.DPMLM",
    texts: Sequence[str],
    epsilon: float,
    max_tokens: int | None = 256,
) -> List[str]:
    """
    Rewrite a batch of texts with DP-MLM at a fixed epsilon.
    """
    outputs: List[str] = []
    for text in texts:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = _truncate(text, max_tokens=max_tokens)

        priv = dp.dpmlm_rewrite(
            sentence=text,
            epsilon=epsilon,
            REPLACE=False,
            FILTER=False,
            STOP=False,
            TEMP=True,
            POS=True,
            CONCAT=True,
        )
        outputs.append(priv[0] if isinstance(priv, (list, tuple)) else priv)
    return outputs


def length_ratio(original: str, privatized: str) -> float:
    """Return (#words in privatized) / (#words in original)."""
    orig_words = len((original or "").split())
    priv_words = len((privatized or "").split())
    return priv_words / orig_words if orig_words else math.nan


def token_change_fraction(original: str, privatized: str) -> float:
    """
    Approximate '% tokens changed' between original and privatized text.
    """
    orig_tokens = (original or "").split()
    priv_tokens = (privatized or "").split()
    max_len = max(len(orig_tokens), len(priv_tokens))
    if max_len == 0:
        return math.nan

    same = sum(
        1 for o, p in zip(orig_tokens, priv_tokens) if o == p
    )
    return 1.0 - (same / max_len)
