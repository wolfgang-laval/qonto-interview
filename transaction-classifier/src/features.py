import os
import pandas as pd
import numpy as np
import re

from src.config import (
    RAW_FILENAME,
    PROCESSED_FILENAME,
    TEXT_COL,
    NUMERIC_COLS,
    CATEGORIC_COLS,
    TARGET,
)

_NOISE_PATTERNS = re.compile(
    r"""
    # ── Identifiants techniques ───────────────────────────────────────────────
    \b[A-Z0-9]{15,}\b               |  # Refs SEPA / identifiants longs
    \b\d{5,}\b                      |  # Suites de chiffres (numéros de transaction)
    /[A-Z]{2,4}/                    |  # Balises SEPA type /PID/ /IID/ /SDT/ /EID/
    [A-Z]{2,4}-\d{2,}               |  # Codes comme FA-24, EPC30B5

    # ── Libellés virements & prélèvements ────────────────────────────────────
    PRLV\s+SEPA                     |
    PRLV                            |
    VIR\s+INST                      |
    VIR\b                           |
    VIREMENT\s+SEPA                 |
    VIREMENT                        |
    SEPA\s+EMIS                     |
    SEPA                            |
    EFFETS?\s+DOMICILIES?           |
    DOMICILIES?                     |
    EUROPEEN\s+EMIS\s+NET\s+POUR    |  # "europeen emis net pour:"
    EMIS\s+NET\s+POUR               |
    EMIS\b                          |
    NET\s+POUR                      |

    # ── Champs de référence bancaire ──────────────────────────────────────────
    \bREF\s*:\s*                    |
    \bREMISE\s*:\s*                 |
    \bMOTIF\s*:\s*                  |
    \bCODE\s+ADH\b                  |
    \bFACT\b                        |
    \bRNF\b                         |
    \bRMT\b                         |

    # ── Termes génériques cartes & paiements ──────────────────────────────────
    CARTE\s+BLEUE                   |
    CARTE\s+[A-Z]+\s+-              |  # "carte majid -"
    \bCARTE\b                       |
    \bCHEQUE\s+N\b                  |
    \bCHEQUE\b                      |

    # ── Bruit de formatage ────────────────────────────────────────────────────
    \s*-\s*-\s*                     |  # Doubles tirets "- -" ou " - - "
    \s+-\s*$                        |  # Tiret en fin de chaîne
    ^\s*-\s+                        |  # Tiret en début de chaîne
    [/#@|\\]                        |  # Caractères parasites
    \s{2,}                             # Espaces multiples
    """,
    re.VERBOSE | re.IGNORECASE,
)

_AMOUNT_BINS = [0, 50, 500, 5_000, np.inf]
_AMOUNT_LABELS = [0, 1, 2, 3]


def _clean_text(text: str) -> str:
    """Normalize banking text"""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = _NOISE_PATTERNS.sub(" ", text)
    text = text.strip().strip("-").strip()
    return text


def _build_text_feature(row) -> str:
    """
    Concatenate MERCHANT_NAME and DESCRIPTION into a single text field,
    after deduplication of redundant tokens.
    """
    merchant = _clean_text(row["MERCHANT_NAME"])
    description = _clean_text(row["DESCRIPTION"])

    if not description:
        return merchant

    merchant_tokens = set(t for t in merchant.split() if len(t) > 2)

    new_tokens = [
        t for t in description.split() if t not in merchant_tokens and len(t) > 2
    ]

    if len(new_tokens) >= 2:
        return f"{merchant} {' '.join(new_tokens)}".strip()

    return merchant


def _month_third(day: int) -> int:
    """
    Split the month into 3 thirds.
    1 = beginning (1-10), 2 = middle (11-20),
    """
    if day <= 10:
        return 1
    elif day <= 20:
        return 2
    return 3


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set from the raw DataFrame.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["DATE_EMITTED"]):
        df["DATE_EMITTED"] = pd.to_datetime(df["DATE_EMITTED"])

    df["TEXT"] = df.apply(_build_text_feature, axis=1)

    abs_amount = df["AMOUNT"].abs()

    df["LOG_AMOUNT"] = np.log1p(abs_amount)
    df["AMOUNT_BUCKET"] = pd.cut(
        abs_amount,
        bins=_AMOUNT_BINS,
        labels=_AMOUNT_LABELS,
        right=True,
    ).astype(int)

    df["TYPE_OF_PAYMENT"] = df["TYPE_OF_PAYMENT"].fillna("Unknown").str.strip()
    df["DAY_OF_WEEK"] = df["DATE_EMITTED"].dt.dayofweek
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)
    df["DAY_OF_MONTH"] = df["DATE_EMITTED"].dt.day
    df["MONTH_THIRD"] = df["DAY_OF_MONTH"].apply(_month_third)
    df["MONTH"] = df["DATE_EMITTED"].dt.month
    df["IS_DESCRIPTION_EMPTY"] = (
        df["DESCRIPTION"].isna() | (df["DESCRIPTION"].str.strip() == "")
    ).astype(int)
    df["TEXT_LENGTH"] = df["TEXT"].str.len()

    """
    counts = df[TARGET].value_counts()
    rare = counts[counts < MIN_SAMPLES].index.tolist()
    if rare:
        print(f"Removing categories (< {MIN_SAMPLES} ex.) : {rare}")
        df = df[~df[TARGET].isin(rare)].reset_index(drop=True)
    """
    output_cols = TEXT_COL + NUMERIC_COLS + CATEGORIC_COLS + TARGET
    output_cols.append("DATE_EMITTED")  # for temporal split in evaluate.py

    return df[output_cols]


if __name__ == "__main__":
    current_dir = os.getcwd()
    raw_file = os.path.join(current_dir, "data", "raw", RAW_FILENAME)
    clean_file = os.path.join(current_dir, "data", "processed", PROCESSED_FILENAME)

    raw = pd.read_csv(raw_file)
    features = build_features(raw)
    features.to_csv(clean_file, index=False)
