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


def match_span_predictions(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match spans one-to-one.

    A pair is eligible when both spans have the same ``clause_type`` and their
    character-span IoU is at least ``iou_threshold``. Eligible pairs are matched
    highest-IoU first, with stable ties by lower gold index then lower prediction
    index. Each gold span and each prediction can participate in at most one match.
    """
    eligible: list[tuple[float, int, int]] = []
    for gold_index, gold_span in enumerate(gold):
        for pred_index, prediction in enumerate(predictions):
            if gold_span["clause_type"] != prediction["clause_type"]:
                continue
            iou = span_iou_score(
                (prediction["char_start"], prediction["char_end"]),
                (gold_span["char_start"], gold_span["char_end"]),
            )
            if iou >= iou_threshold:
                eligible.append((iou, gold_index, pred_index))

    eligible.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gold_index, pred_index in eligible:
        if gold_index in matched_gold or pred_index in matched_predictions:
            continue
        matched_gold.add(gold_index)
        matched_predictions.add(pred_index)
        matches.append((gold_index, pred_index, iou))

    return {
        "matches": matches,
        "unmatched_gold": [index for index in range(len(gold)) if index not in matched_gold],
        "unmatched_predictions": [
            index for index in range(len(predictions)) if index not in matched_predictions
        ],
    }


def compute_span_detection_metrics(
    gold_by_contract: dict[str, list[dict[str, Any]]],
    predictions_by_contract: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute prediction-level span-detection precision, recall, and F1.

    Matching is one-to-one by clause type and span IoU (see
    :func:`match_span_predictions`). Every unmatched prediction is a false
    positive, including duplicates and predictions with the wrong clause type.
    Candidate coverage is type-agnostic: a gold span is covered when any
    predicted span reaches ``iou_threshold``, so extraction misses remain visible
    even when the classifier assigns the wrong type.
    """
    labels = sorted(
        {span["clause_type"] for spans in gold_by_contract.values() for span in spans}
    )
    stats = {label: {"tp": 0, "fp": 0, "fn": 0, "covered": 0, "overlapped": 0} for label in labels}
    total_tp = total_fp = total_fn = total_covered = total_overlapped = 0
    gold_count = sum(len(spans) for spans in gold_by_contract.values())
    prediction_count = sum(len(spans) for spans in predictions_by_contract.values())

    contract_ids = set(gold_by_contract) | set(predictions_by_contract)
    for contract_id in contract_ids:
        gold = gold_by_contract.get(contract_id, [])
        predictions = predictions_by_contract.get(contract_id, [])
        matching = match_span_predictions(gold, predictions, iou_threshold)

        matched_gold = {gold_index for gold_index, _, _ in matching["matches"]}
        matched_predictions = {pred_index for _, pred_index, _ in matching["matches"]}
        for gold_index, prediction_index, _ in matching["matches"]:
            label = gold[gold_index]["clause_type"]
            stats[label]["tp"] += 1
            total_tp += 1

        for gold_index, gold_span in enumerate(gold):
            label = gold_span["clause_type"]
            covered = any(
                span_iou_score(
                    (prediction["char_start"], prediction["char_end"]),
                    (gold_span["char_start"], gold_span["char_end"]),
                )
                >= iou_threshold
                for prediction in predictions
            )
            overlapped = any(
                min(prediction["char_end"], gold_span["char_end"])
                > max(prediction["char_start"], gold_span["char_start"])
                for prediction in predictions
            )
            if covered:
                stats[label]["covered"] += 1
                total_covered += 1
            if overlapped:
                stats[label]["overlapped"] += 1
                total_overlapped += 1
            if gold_index not in matched_gold:
                stats[label]["fn"] += 1
                total_fn += 1

        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions:
                continue
            label = prediction["clause_type"]
            if label in stats:
                stats[label]["fp"] += 1
            total_fp += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = tp + fn
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "candidate_coverage": round(counts["covered"] / support, 4) if support else 0.0,
            "candidate_overlap_recall": round(counts["overlapped"] / support, 4) if support else 0.0,
        }
        f1s.append(f1)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "matching_rule": (
            "one-to-one greedy matching; same clause_type; span IoU >= "
            f"{iou_threshold:.2f}; highest IoU first; ties by lower gold then prediction index"
        ),
        "iou_threshold": iou_threshold,
        "gold_spans": gold_count,
        "predicted_spans": prediction_count,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "candidate_coverage": round(total_covered / gold_count, 4) if gold_count else 0.0,
        "candidate_overlap_recall": round(total_overlapped / gold_count, 4) if gold_count else 0.0,
        "per_class": per_class,
    }
