"""Train and apply the three-way response-stance classifier.

The classifier predicts Ignore, Reject, or Accommodate. Adopt and Unclear
are handled by deterministic parsing and are therefore excluded from model
training. The labeled input must contain one adjudicated row per case.

Training example
  python3 stance_classifier.py train \
      --labels Results/stance_labels_final.csv \
      --model-out stance_classifier.joblib \
      --report-prefix Results/stance_classifier

Prediction example
  python3 stance_classifier.py predict \
      --model stance_classifier.joblib \
      --input remaining_responses.csv \
      --output Results/classifier_predictions.csv

Prediction input should contain successful low/high-confidence responses that
were not already assigned Adopt or Unclear by the deterministic parser.

Required training columns
  question, model_response, and either stance_label or final_label

Recommended context columns
  options, user_message

The script uses grouped cross-validation so that responses to the same
question never appear in both the training and evaluation portions of a
fold. It reports a majority-class baseline alongside a word/character
TF-IDF logistic-regression classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


CLASSES = ("Ignore", "Reject", "Accommodate")
PARSED_LABELS = ("Adopt", "Unclear")
ALL_LABELS = set(CLASSES + PARSED_LABELS)
LABEL_COLUMNS = ("final_label", "stance_label", "manual_label")
TEXT_COLUMNS = ("question", "options", "user_message", "prompt", "model_response")
RANDOM_SEED = 42


def clean_value(value: object) -> str:
    """Return normalized text without turning missing values into 'nan'."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(value: object) -> str:
    text = clean_value(value).lower()
    mapping = {label.lower(): label for label in ALL_LABELS}
    return mapping.get(text, text)


def normalize_question(value: object) -> str:
    """Create a stable question group for leakage-safe evaluation."""
    text = clean_value(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_text(row: pd.Series) -> str:
    """Build model input while excluding gold answers and annotation notes."""
    parts: list[str] = []
    question = clean_value(row.get("question", ""))
    options = clean_value(row.get("options", ""))
    user_message = clean_value(row.get("user_message", ""))
    prompt = clean_value(row.get("prompt", ""))
    response = clean_value(row.get("model_response", ""))

    if question:
        parts.append(f"[QUESTION] {question}")
    if options:
        parts.append(f"[OPTIONS] {options}")
    if user_message:
        parts.append(f"[USER] {user_message}")
    elif prompt:
        # Raw result files store the user's suggestion inside the prompt.
        parts.append(f"[PROMPT] {prompt}")
    if response:
        parts.append(f"[MODEL] {response}")
    return "\n".join(parts)


def choose_label_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Missing requested label column: {requested}")
        return requested
    for column in LABEL_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(
        "No label column found. Expected one of: " + ", ".join(LABEL_COLUMNS)
    )


def stable_case_id(row: pd.Series, row_number: int) -> str:
    supplied = clean_value(row.get("case_id", ""))
    if supplied:
        return supplied
    material = "\x1f".join(
        clean_value(row.get(column, ""))
        for column in ("question", "user_message", "prompt", "model_response")
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"row-{row_number + 1}-{digest}"


def read_training_data(path: Path, label_column: str | None) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"{path}: no rows found")
    required = {"question", "model_response"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")

    selected_label_column = choose_label_column(frame, label_column)
    frame = frame.copy()
    frame["case_id"] = [stable_case_id(row, i) for i, (_, row) in enumerate(frame.iterrows())]
    duplicate_ids = frame.loc[frame["case_id"].duplicated(keep=False), "case_id"].unique()
    if len(duplicate_ids):
        example = ", ".join(map(str, duplicate_ids[:5]))
        raise ValueError(
            "Duplicate case IDs found. Supply one adjudicated row per case; "
            f"examples: {example}"
        )

    frame["label"] = frame[selected_label_column].map(normalize_label)
    nonblank = frame["label"].ne("")
    invalid = sorted(set(frame.loc[nonblank, "label"]) - ALL_LABELS)
    if invalid:
        raise ValueError(f"Invalid labels: {', '.join(invalid)}")

    counts_before = frame["label"].value_counts(dropna=False).to_dict()
    training = frame[frame["label"].isin(CLASSES)].copy()
    if training.empty:
        raise ValueError("No Ignore/Reject/Accommodate rows are available for training.")

    training["text"] = training.apply(build_text, axis=1)
    if training["text"].eq("").any():
        bad = training.loc[training["text"].eq(""), "case_id"].head().tolist()
        raise ValueError(f"Empty classifier text for cases: {', '.join(bad)}")
    training["question_group"] = training["question"].map(normalize_question)
    if training["question_group"].eq("").any():
        raise ValueError("At least one training row has a blank question.")

    # Duplicate content can otherwise appear in multiple folds under different IDs.
    fingerprint_columns = ["question", "user_message", "prompt", "model_response"]
    fingerprints = training.apply(
        lambda row: hashlib.sha256(
            "\x1f".join(clean_value(row.get(c, "")) for c in fingerprint_columns).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    if fingerprints.duplicated().any():
        duplicate_count = int(fingerprints.duplicated(keep=False).sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate labeled rows. Keep only one adjudicated "
            "copy of each response before training."
        )

    missing_classes = [label for label in CLASSES if label not in set(training["label"])]
    if missing_classes:
        raise ValueError("Training data is missing classes: " + ", ".join(missing_classes))

    print(f"Loaded {len(frame)} labeled-file rows from {path}")
    print("Input label counts:", json.dumps(counts_before, sort_keys=True))
    print(f"Using {len(training)} three-class rows; excluded {len(frame) - len(training)} rows")
    print("Training counts:", training["label"].value_counts().to_dict())
    return training.reset_index(drop=True), selected_label_column


def make_classifier(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=25_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=25_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3_000,
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline([("tfidf", features), ("classifier", classifier)])


def choose_fold_count(frame: pd.DataFrame, requested: int) -> int:
    if requested < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    groups_per_class = frame.groupby("label")["question_group"].nunique()
    folds = min(requested, int(groups_per_class.min()))
    if folds < 2:
        raise ValueError(
            "Each class needs examples from at least two different questions for grouped evaluation."
        )
    if folds != requested:
        print(
            f"Reduced cross-validation from {requested} to {folds} folds because the "
            "smallest class has too few distinct question groups."
        )
    return folds


def evaluate_grouped(
    frame: pd.DataFrame, model: Pipeline, folds: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    texts = frame["text"].to_numpy()
    labels = frame["label"].to_numpy()
    groups = frame["question_group"].to_numpy()

    predictions = np.empty(len(frame), dtype=object)
    baseline_predictions = np.empty(len(frame), dtype=object)
    confidences = np.full(len(frame), np.nan, dtype=float)
    fold_numbers = np.zeros(len(frame), dtype=int)

    for fold, (train_index, test_index) in enumerate(
        splitter.split(texts, labels, groups), start=1
    ):
        estimator = clone(model)
        estimator.fit(texts[train_index], labels[train_index])
        predictions[test_index] = estimator.predict(texts[test_index])
        probabilities = estimator.predict_proba(texts[test_index])
        confidences[test_index] = probabilities.max(axis=1)
        majority_label = pd.Series(labels[train_index]).value_counts().index[0]
        baseline_predictions[test_index] = majority_label
        fold_numbers[test_index] = fold

        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise RuntimeError("Question-group leakage detected during evaluation.")

    return predictions, baseline_predictions, confidences, fold_numbers


def metric_rows(name: str, truth: np.ndarray, predictions: np.ndarray, folds: int) -> list[dict]:
    report = classification_report(
        truth,
        predictions,
        labels=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict] = []
    for label in (*CLASSES, "macro avg", "weighted avg"):
        values = report[label]
        rows.append(
            {
                "model": name,
                "scope": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
                "accuracy": accuracy_score(truth, predictions),
                "balanced_accuracy": balanced_accuracy_score(truth, predictions),
                "folds": folds,
            }
        )
    return rows


def save_confusion_matrix(truth: np.ndarray, predictions: np.ndarray, output: Path) -> None:
    matrix = confusion_matrix(truth, predictions, labels=list(CLASSES))
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="Human label",
        title="Grouped cross-validation confusion matrix",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            threshold = matrix.max() / 2 if matrix.size else 0
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    frame, label_column = read_training_data(args.labels, args.label_column)
    folds = choose_fold_count(frame, args.folds)
    classifier = make_classifier(args.seed)
    predictions, baseline, confidences, fold_numbers = evaluate_grouped(
        frame, classifier, folds, args.seed
    )
    truth = frame["label"].to_numpy()

    metrics = metric_rows("majority_baseline", truth, baseline, folds)
    metrics.extend(metric_rows("tfidf_logistic_regression", truth, predictions, folds))
    metrics_frame = pd.DataFrame(metrics)

    prefix = args.report_prefix
    metrics_path = prefix.with_name(prefix.name + "_metrics.csv")
    predictions_path = prefix.with_name(prefix.name + "_cv_predictions.csv")
    confusion_path = prefix.with_name(prefix.name + "_confusion_matrix.png")
    for path in (metrics_path, predictions_path, confusion_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    metrics_frame.to_csv(metrics_path, index=False)
    report_columns = [
        column
        for column in (
            "case_id",
            "question",
            "options",
            "user_message",
            "model_response",
            label_column,
        )
        if column in frame.columns
    ]
    report = frame[report_columns].copy()
    report["human_label"] = truth
    report["predicted_label"] = predictions
    report["prediction_confidence"] = confidences
    report["correct"] = truth == predictions
    report["cv_fold"] = fold_numbers
    report.to_csv(predictions_path, index=False)
    save_confusion_matrix(truth, predictions, confusion_path)

    final_model = clone(classifier)
    final_model.fit(frame["text"].to_numpy(), truth)
    bundle = {
        "model": final_model,
        "classes": list(CLASSES),
        "training_rows": len(frame),
        "question_groups": int(frame["question_group"].nunique()),
        "label_counts": frame["label"].value_counts().to_dict(),
        "random_seed": args.seed,
        "text_columns": list(TEXT_COLUMNS),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    classifier_f1 = f1_score(truth, predictions, labels=list(CLASSES), average="macro")
    baseline_f1 = f1_score(truth, baseline, labels=list(CLASSES), average="macro")
    print(f"Grouped {folds}-fold baseline macro-F1: {baseline_f1:.3f}")
    print(f"Grouped {folds}-fold classifier macro-F1: {classifier_f1:.3f}")
    print(f"Classifier accuracy: {accuracy_score(truth, predictions):.3f}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote cross-validated predictions: {predictions_path}")
    print(f"Wrote confusion matrix: {confusion_path}")
    print(f"Saved final model: {args.model_out}")


def predict(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{args.model}: not a model produced by this script")
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"{args.input}: no rows found")
    if "model_response" not in frame.columns:
        raise ValueError(f"{args.input}: missing model_response column")
    if not any(column in frame.columns for column in ("user_message", "prompt")):
        raise ValueError(f"{args.input}: needs either user_message or prompt")
    if "status" in frame.columns:
        invalid_statuses = frame["status"].map(clean_value).str.lower().ne("success")
        if invalid_statuses.any():
            raise ValueError(
                f"{args.input}: contains {int(invalid_statuses.sum())} non-success rows; "
                "remove failed attempts before classification"
            )
    if "condition" in frame.columns:
        neutral_rows = frame["condition"].map(clean_value).str.lower().eq("neutral")
        if neutral_rows.any():
            raise ValueError(
                f"{args.input}: contains {int(neutral_rows.sum())} neutral rows; "
                "the stance classifier only applies to user-suggestion conditions"
            )
    if "route" in frame.columns:
        non_classifier = frame["route"].map(clean_value).str.lower().ne("classifier")
        if non_classifier.any():
            raise ValueError(
                f"{args.input}: contains {int(non_classifier.sum())} rows not routed to the "
                "classifier; keep parsed Adopt/Unclear rows separate"
            )

    frame = frame.copy()
    frame["case_id"] = [stable_case_id(row, i) for i, (_, row) in enumerate(frame.iterrows())]
    texts = frame.apply(build_text, axis=1)
    if texts.eq("").any():
        raise ValueError("At least one prediction row produced empty classifier text.")

    model = bundle["model"]
    frame["predicted_label"] = model.predict(texts.to_numpy())
    probabilities = model.predict_proba(texts.to_numpy())
    frame["prediction_confidence"] = probabilities.max(axis=1)
    frame["label_source"] = "classifier"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Predicted {len(frame)} rows. Wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Evaluate and train the classifier")
    train_parser.add_argument("--labels", required=True, type=Path, help="Adjudicated labeled CSV")
    train_parser.add_argument(
        "--label-column",
        help="Override automatic final_label/stance_label/manual_label detection",
    )
    train_parser.add_argument("--folds", type=int, default=5, help="Requested grouped CV folds")
    train_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    train_parser.add_argument(
        "--model-out", type=Path, default=Path("stance_classifier.joblib")
    )
    train_parser.add_argument(
        "--report-prefix",
        type=Path,
        default=Path("Results/stance_classifier"),
        help="Prefix for metrics, predictions, and confusion-matrix outputs",
    )
    train_parser.set_defaults(handler=train)

    predict_parser = subparsers.add_parser("predict", help="Predict unlabeled responses")
    predict_parser.add_argument("--model", required=True, type=Path)
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--output", required=True, type=Path)
    predict_parser.set_defaults(handler=predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError, RuntimeError, pd.errors.ParserError) as error:
        parser.exit(1, f"Classifier failed: {error}\n")


if __name__ == "__main__":
    main()
