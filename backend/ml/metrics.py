"""Evaluation metrics for clause classification and span extraction."""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Any


def normalize_text_for_eval(text: str) -> str:
    """Normalize text for evaluation: lowercasing, removing punctuation and extra whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return " ".join(text.split())


def exact_match_score(pred: str | None, gold: str | None) -> float:
    """Exact Match (EM): returns 1.0 if normalized strings match, 0.0 otherwise."""
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    return 1.0 if normalize_text_for_eval(pred) == normalize_text_for_eval(gold) else 0.0


def token_f1_score(pred: str | None, gold: str | None) -> float:
    """Compute token-level F1 between predicted and gold text spans."""
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0

    pred_tokens = normalize_text_for_eval(pred).split()
    gold_tokens = normalize_text_for_eval(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    # Count overlaps
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    num_same = sum(min(pred_counts[token], gold_counts[token]) for token in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def span_iou_score(
    pred_offsets: tuple[int, int] | None,
    gold_offsets: tuple[int, int] | None,
) -> float:
    """Compute character-level Intersection over Union (Span-IoU).

    span_iou = |pred ∩ gold| / |pred ∪ gold|
    """
    if pred_offsets is None and gold_offsets is None:
        return 1.0
    if pred_offsets is None or gold_offsets is None:
        return 0.0

    p_start, p_end = pred_offsets
    g_start, g_end = gold_offsets

    if p_start >= p_end and g_start >= g_end:
        return 1.0
    if p_start >= p_end or g_start >= g_end:
        return 0.0

    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    intersection = max(0, inter_end - inter_start)

    union = (p_end - p_start) + (g_end - g_start) - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute precision, recall, and F1 scores per clause type and macro average."""
    if not y_true:
        return {"macro_f1": 0.0, "per_class": {}}

    unique_labels = sorted(set(y_true) | set(y_pred)) if labels is None else labels
    per_class: dict[str, dict[str, float]] = {}

    f1s: list[float] = []

    for label in unique_labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if yt == label),
        }
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }
