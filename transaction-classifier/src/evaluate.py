import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, case: str
) -> None:
    """Evaluate the model on the test set and
    save the results (classification report,
    confusion matrix, confidence distribution)."""

    current_dir = os.getcwd()

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    macro_f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=0)[
        "macro avg"
    ]["f1-score"]
    print(f"{case} macro F1-score: {macro_f1:.3f}")

    figures_path = os.path.join(current_dir, "figures", "models")

    # ── 1. Classification report
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T
    report_df.to_csv(
        os.path.join(figures_path, f"{case}_classification_report.csv"),
        float_format="%.3f",
    )

    # ── 2. Normalized confusion matrix
    labels = sorted(y_test.unique())
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) * 0.8)))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=labels,
        normalize="true",
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(
        "Normalized confusion matrix (recall per class)", fontsize=13, fontweight="bold"
    )
    fig.savefig(
        os.path.join(figures_path, f"{case}_confusion_matrix.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)

    # ── 3. Distribution de la confiance (max proba)
    max_proba = y_proba.max(axis=1)
    correct = (y_pred == y_test).values

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(max_proba[correct], bins=40, alpha=0.6, color="#5C3BFE", label="Correct")
    ax.hist(max_proba[~correct], bins=40, alpha=0.6, color="#FF6B6B", label="Incorrect")
    ax.set_title(
        "Confidence distribution (max predicted probability)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Number of transactions")
    ax.legend()
    fig.savefig(
        os.path.join(figures_path, f"{case}_confidence_distribution.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)
