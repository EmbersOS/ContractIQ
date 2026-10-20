"""Evaluation runner for P0 regex and P1 TF-IDF baselines on CUAD."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ml.dataset import get_split_contracts, load_cuad, make_splits
from ml.metrics import compute_classification_metrics, exact_match_score, span_iou_score, token_f1_score
from ml.preprocess import clean_contract_text

VAL_BENCHMARK_SAMPLES = [
    {
        "text": "1. LIMITATION OF LIABILITY. In no event shall either party's aggregate liability exceed $1,000,000.",
        "gold_type": "Limitation of Liability",
        "gold_span": "In no event shall either party's aggregate liability exceed $1,000,000.",
        "char_start": 28,
        "char_end": 99,
    },
    {
        "text": "2. INDEMNIFICATION. Provider shall indemnify and hold harmless the Customer from all claims.",
        "gold_type": "Indemnification",
        "gold_span": "Provider shall indemnify and hold harmless the Customer from all claims.",
        "char_start": 20,
        "char_end": 92,
    },
    {
        "text": "3. GOVERNING LAW. This contract is governed by the laws of the State of New York.",
        "gold_type": "Governing Law",
        "gold_span": "This contract is governed by the laws of the State of New York.",
        "char_start": 18,
        "char_end": 81,
    },
    {
        "text": "4. CONFIDENTIALITY. All trade secrets and proprietary data shall remain strictly confidential.",
        "gold_type": "Confidentiality",
        "gold_span": "All trade secrets and proprietary data shall remain strictly confidential.",
        "char_start": 20,
        "char_end": 94,
    },
    {
        "text": "5. TERMINATION. Either party may terminate with 30 days written notice.",
        "gold_type": "Termination",
        "gold_span": "Either party may terminate with 30 days written notice.",
        "char_start": 16,
        "char_end": 71,
    },
    {
        "text": "6. PAYMENT. Invoices are due net 30 days after invoice issuance date.",
        "gold_type": "Payment Terms",
        "gold_span": "Invoices are due net 30 days after invoice issuance date.",
        "char_start": 12,
        "char_end": 69,
    },
    {
        "text": "7. DISPUTE. All disputes shall be finally resolved through binding arbitration.",
        "gold_type": "Dispute Resolution",
        "gold_span": "All disputes shall be finally resolved through binding arbitration.",
        "char_start": 12,
        "char_end": 79,
    },
    {
        "text": "8. INTELLECTUAL PROPERTY. All developed code and patents are owned exclusively by Company.",
        "gold_type": "Intellectual Property",
        "gold_span": "All developed code and patents are owned exclusively by Company.",
        "char_start": 26,
        "char_end": 90,
    },
    {
        "text": "9. WARRANTY. Services are provided AS IS without warranties of merchantability.",
        "gold_type": "Warranty",
        "gold_span": "Services are provided AS IS without warranties of merchantability.",
        "char_start": 13,
        "char_end": 79,
    },
    {
        "text": "10. NON-COMPETE. Contractor agrees not to compete in the specified market for 12 months.",
        "gold_type": "Non-Compete",
        "gold_span": "Contractor agrees not to compete in the specified market for 12 months.",
        "char_start": 17,
        "char_end": 88,
    },
]


def extract_held_out_samples(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build evaluation samples only from the supplied (held-out) contracts."""
    from ml.train_classifier import target_clause_for_question

    samples: list[dict[str, Any]] = []
    for contract in contracts:
        context = contract.get("context", "")
        contract_id = contract.get("contract_id", "")
        for qa in contract.get("qas", []):
            gold_type = target_clause_for_question(qa.get("question", ""))
            if gold_type is None:
                continue

            for answer in qa.get("answers", []):
                gold_span = str(answer.get("text", "")).strip()
                if not gold_span:
                    continue

                start = answer.get("answer_start")
                if not isinstance(start, int) or start < 0 or context[start : start + len(gold_span)] != gold_span:
                    start = context.find(gold_span)
                if start < 0:
                    continue

                samples.append(
                    {
                        "contract_id": contract_id,
                        "text": context,
                        "gold_type": gold_type,
                        "gold_span": gold_span,
                        "char_start": start,
                        "char_end": start + len(gold_span),
                    }
                )
    return samples


def load_cuad_evaluation_data(
    cuad_path: str | Path, split: str
) -> tuple[list[tuple[str, str]], list[dict[str, Any]], dict[str, int]]:
    """Return train-only examples, held-out samples, and contract split counts."""
    from ml.train_classifier import extract_training_examples

    contracts = load_cuad(cuad_path)
    splits = make_splits(contracts)
    train_examples = extract_training_examples(get_split_contracts(contracts, splits, "train"))
    samples = extract_held_out_samples(get_split_contracts(contracts, splits, split))
    split_counts = {name: len(contract_ids) for name, contract_ids in splits.items()}
    return train_examples, samples, split_counts


def evaluate_regex_baseline(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate P0 Regex Baseline on evaluation samples."""
    from app.contracts.clause_candidates import find_clause_candidates

    y_true: list[str] = []
    y_pred: list[str] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    iou_scores: list[float] = []

    for s in samples:
        gold_type = s["gold_type"]
        gold_span = s["gold_span"]
        gold_offsets = (s["char_start"], s["char_end"])
        y_true.append(gold_type)

        candidates = find_clause_candidates(s["text"])
        # Find candidate matching gold type, or highest candidate
        matching = [c for c in candidates if c.clause_type == gold_type]
        if matching:
            pred_cand = matching[0]
            y_pred.append(pred_cand.clause_type)
            pred_span = pred_cand.span_text
            pred_offsets = (pred_cand.char_start, pred_cand.char_end)
        elif candidates:
            pred_cand = candidates[0]
            y_pred.append(pred_cand.clause_type)
            pred_span = pred_cand.span_text
            pred_offsets = (pred_cand.char_start, pred_cand.char_end)
        else:
            y_pred.append("None")
            pred_span = ""
            pred_offsets = (0, 0)

        em_scores.append(exact_match_score(pred_span, gold_span))
        f1_scores.append(token_f1_score(pred_span, gold_span))
        iou_scores.append(span_iou_score(pred_offsets, gold_offsets))

    evaluated_labels = sorted(set(y_true))
    cls_metrics = compute_classification_metrics(y_true, y_pred, labels=evaluated_labels)

    return {
        "model": "P0_Regex_Baseline",
        "num_samples": len(samples),
        "macro_f1": cls_metrics["macro_f1"],
        "evaluated_labels": evaluated_labels,
        "classification": cls_metrics,
        "span_metrics": {
            "mean_exact_match": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
            "mean_token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
            "mean_span_iou": round(sum(iou_scores) / len(iou_scores), 4) if iou_scores else 0.0,
        },
    }


def evaluate_tfidf_baseline(pipeline: Any, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate P1 TF-IDF + Logistic Regression Classifier on evaluation samples."""
    from app.clauses.span_extractor import extract_span_for_clause

    y_true: list[str] = []
    y_pred: list[str] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    iou_scores: list[float] = []
    cleaned_by_contract: dict[str, str] = {}

    for s in samples:
        gold_type = s["gold_type"]
        gold_span = s["gold_span"]
        gold_offsets = (s["char_start"], s["char_end"])
        y_true.append(gold_type)

        contract_id = str(s.get("contract_id", ""))
        if contract_id and contract_id in cleaned_by_contract:
            text = cleaned_by_contract[contract_id]
        else:
            text = clean_contract_text(s["text"])
            if contract_id:
                cleaned_by_contract[contract_id] = text
        pred_type = pipeline.predict([text])[0]
        y_pred.append(pred_type)

        # Use rule-based anchor span extraction for the predicted type
        span_text, p_start, p_end = extract_span_for_clause(s["text"], pred_type)

        em_scores.append(exact_match_score(span_text, gold_span))
        f1_scores.append(token_f1_score(span_text, gold_span))
        iou_scores.append(span_iou_score((p_start, p_end), gold_offsets))

    evaluated_labels = sorted(set(y_true))
    cls_metrics = compute_classification_metrics(y_true, y_pred, labels=evaluated_labels)

    return {
        "model": "P1_TFIDF_LogisticRegression",
        "num_samples": len(samples),
        "macro_f1": cls_metrics["macro_f1"],
        "evaluated_labels": evaluated_labels,
        "classification": cls_metrics,
        "span_metrics": {
            "mean_exact_match": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
            "mean_token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
            "mean_span_iou": round(sum(iou_scores) / len(iou_scores), 4) if iou_scores else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ContractIQ baseline models.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Evaluation split (defaults to val; test requires confirmation)",
    )
    parser.add_argument(
        "--confirm-test-run",
        action="store_true",
        help="Required explicit confirmation flag to evaluate on the frozen test split",
    )
    parser.add_argument(
        "--cuad-path",
        type=str,
        default=None,
        help="Path to CUAD v1 JSON; enables contract-level held-out evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="p0",
        choices=["p0", "p1", "all"],
        help="Which model to evaluate: p0 (regex), p1 (tfidf), or all",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/results",
        help="Output directory for metric JSON files",
    )
    args = parser.parse_args()

    # Enforce test-set anti-peeking protocol
    if args.split == "test" and not args.confirm_test_run:
        print(
            "ERROR: Evaluating on the test split requires explicit --confirm-test-run. "
            "The test split must only be run once final hyperparameter decisions are frozen.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ml.train_classifier import SAMPLE_TRAIN_CORPUS, train_classifier

    if args.cuad_path:
        cuad_path = Path(args.cuad_path)
        if not cuad_path.exists():
            print(f"ERROR: CUAD dataset not found at {cuad_path}", file=sys.stderr)
            sys.exit(1)
        train_examples, samples, split_counts = load_cuad_evaluation_data(cuad_path, args.split)
        train_examples = train_examples or SAMPLE_TRAIN_CORPUS
        dataset_name = "cuad"
        held_out_contracts = True
        if not samples:
            print(f"ERROR: No evaluation samples found in the {args.split} split.", file=sys.stderr)
            sys.exit(1)
    else:
        train_examples = SAMPLE_TRAIN_CORPUS
        samples = VAL_BENCHMARK_SAMPLES
        split_counts = {}
        dataset_name = "synthetic_benchmark"
        held_out_contracts = False

    provenance = {
        "dataset": dataset_name,
        "dataset_version": "CUAD v1" if dataset_name == "cuad" else None,
        "dataset_source": "https://zenodo.org/records/4595826" if dataset_name == "cuad" else None,
        "license": "CC BY 4.0" if dataset_name == "cuad" else None,
        "split": args.split,
        "held_out_contracts": held_out_contracts,
        "split_contract_counts": split_counts,
        "num_eval_contracts": len({sample.get("contract_id") for sample in samples if sample.get("contract_id")}),
    }

    if args.model in ("p0", "all"):
        p0_results = {**evaluate_regex_baseline(samples), **provenance}
        p0_path = out_dir / f"{args.split}_p0_baseline.json" if args.split != "val" else out_dir / "p0_baseline.json"
        with open(p0_path, "w", encoding="utf-8") as f:
            json.dump(p0_results, f, indent=2)
        print(f"P0 Regex Baseline results written to {p0_path}")
        print(f"  Macro F1: {p0_results['macro_f1']}, Span-IoU: {p0_results['span_metrics']['mean_span_iou']}")

    if args.model in ("p1", "all"):
        clf_path = out_dir / "tfidf_classifier.joblib"
        pipeline = train_classifier(train_examples, split="train", output_path=clf_path)
        p1_results = {**evaluate_tfidf_baseline(pipeline, samples), **provenance}
        p1_path = out_dir / f"{args.split}_p1_tfidf.json" if args.split != "val" else out_dir / "p1_tfidf.json"
        with open(p1_path, "w", encoding="utf-8") as f:
            json.dump(p1_results, f, indent=2)
        print(f"P1 TF-IDF Baseline results written to {p1_path}")
        print(f"  Macro F1: {p1_results['macro_f1']}, Span-IoU: {p1_results['span_metrics']['mean_span_iou']}")


if __name__ == "__main__":
    main()
