import joblib
import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from src.features import build_features
from src.evaluate import evaluate
from src.config import (
    RAW_FILENAME,
    TEXT_COL,
    NUMERIC_COLS,
    CATEGORIC_COLS,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
)

current_dir = os.getcwd()


def temporal_split(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by DATE_EMITTED and split into train/test sets.
    """
    df_sorted = df.sort_values("DATE_EMITTED").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()

    return train, test


def build_pipeline() -> Pipeline:
    """
    Build the machine learning pipeline with preprocessing and classifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(max_features=2000),
                TEXT_COL[0],
            ),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORIC_COLS,
            ),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",
    )

    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", lgbm_clf),
        ]
    )

    return pipeline


def train() -> None:

    # 1. Load data
    print("[1/6] Loading data...")
    data_path = os.path.join(current_dir, "data", "raw", RAW_FILENAME)
    raw = pd.read_csv(data_path, parse_dates=["DATE_EMITTED"])

    # 2. Feature engineering
    print("[2/6] Feature engineering...")
    features = build_features(raw)

    # 3. Temporal split (train/test)
    print("[3/6] Temporal split...")
    train_df, test_df = temporal_split(features, TEST_SIZE)
    X_train = train_df.drop(columns=[TARGET[0], "DATE_EMITTED"])
    y_train = train_df[TARGET[0]]
    X_test = test_df.drop(columns=[TARGET[0], "DATE_EMITTED"])
    y_test = test_df[TARGET[0]]

    # 4. Entraînement
    print("[4/6] Training...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Save model
    print("[5/6] Saving model...")
    model_path = os.path.join(current_dir, "models", "baseline_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    # 6. Evaluation
    print("[6/6] Evaluating...")
    evaluate(pipeline, X_test, y_test, "baseline")


if __name__ == "__main__":
    train()
