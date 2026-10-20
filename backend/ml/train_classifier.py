"""Train TF-IDF + Logistic Regression clause classifier with anti-leakage protection."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from ml.dataset import get_split_contracts, load_cuad, make_splits
from ml.preprocess import GuardedSplitContext, clean_contract_text

# Target clause categories
TARGET_CLAUSES = [
    "Limitation of Liability",
    "Indemnification",
    "Termination",
    "Governing Law",
    "Confidentiality",
    "Non-Compete",
    "Warranty",
    "Intellectual Property",
    "Payment Terms",
    "Dispute Resolution",
]

# Explicit mapping from official CUAD v1 labels to this project's clause categories.
# Categories without a faithful CUAD equivalent are intentionally not mapped.
CUAD_CATEGORY_TO_TARGET = {
    "Cap On Liability": "Limitation of Liability",
    "Uncapped Liability": "Limitation of Liability",
    "Termination For Convenience": "Termination",
    "Governing Law": "Governing Law",
    "Non-Compete": "Non-Compete",
    "Warranty Duration": "Warranty",
    "Ip Ownership Assignment": "Intellectual Property",
}

_CUAD_QUESTION_RE = re.compile(r'related to "([^"]+)"')


def target_clause_for_question(question: str) -> str | None:
    """Return the mapped target category for an official CUAD question."""
    match = _CUAD_QUESTION_RE.search(question)
    if match is None:
        return None
    return CUAD_CATEGORY_TO_TARGET.get(match.group(1))


# Seed corpus for training/testing when offline or CUAD raw file is not present
SAMPLE_TRAIN_CORPUS = [
    ("In no event shall either party's aggregate liability exceed the fees paid under this agreement.", "Limitation of Liability"),
    ("Neither party will be liable for any consequential, indirect, special, or punitive damages.", "Limitation of Liability"),
    ("Company's maximum cumulative liability under this SOW is capped at the total contract value.", "Limitation of Liability"),
    ("Vendor shall indemnify, defend, and hold harmless Customer against any third-party claims or damages.", "Indemnification"),
    ("Each party agrees to indemnify the other from liabilities arising out of gross negligence or willful misconduct.", "Indemnification"),
    ("Either party may terminate this Agreement upon thirty (30) days prior written notice for convenience.", "Termination"),
    ("This contract may be terminated immediately upon material breach following a ten day cure period.", "Termination"),
    ("This Agreement shall be governed by and construed in accordance with the laws of the State of New York.", "Governing Law"),
    ("The parties submit to the exclusive jurisdiction of the state and federal courts located in Delaware.", "Governing Law"),
    ("Each party shall hold proprietary information in strict confidence and prevent unauthorized disclosure.", "Confidentiality"),
    ("Non-disclosure obligations regarding trade secrets shall survive termination for five (5) years.", "Confidentiality"),
    ("During the term and for 12 months thereafter, Consultant shall not engage in competing business activities.", "Non-Compete"),
    ("Employee covenants not to solicit customers or directly compete in the territory.", "Non-Compete"),
    ("Software is provided AS IS with no express or implied warranty of merchantability or fitness for purpose.", "Warranty"),
    ("Provider warrants that the services will be performed in a professional and workmanlike manner.", "Warranty"),
    ("All patents, copyrights, trademarks, and trade secrets developed hereunder belong exclusively to Client.", "Intellectual Property"),
    ("Contractor hereby assigns all right, title, and interest in work product and IP to Company.", "Intellectual Property"),
    ("Payment terms are net 30 days from invoice date. Late payments accrue interest at 1.5% per month.", "Payment Terms"),
    ("Customer shall pay all undisputed invoices within forty-five (45) days of receipt.", "Payment Terms"),
    ("Any controversy or dispute arising out of this contract shall be submitted to binding arbitration.", "Dispute Resolution"),
    ("Disputes shall first undergo good-faith mediation before formal legal proceedings or arbitration.", "Dispute Resolution"),
]


def extract_training_examples(contracts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract (chunk_or_span_text, clause_type) pairs from loaded contracts."""
    examples: list[tuple[str, str]] = []
    for c in contracts:
        for qa in c.get("qas", []):
            matched_type = target_clause_for_question(qa.get("question", ""))
            if not matched_type:
                continue

            answers = qa.get("answers", [])
            for ans in answers:
                text = ans.get("text", "").strip()
                if text:
                    examples.append((text, matched_type))

    return examples


def train_classifier(
    train_data: list[tuple[str, str]],
    split: str = "train",
    output_path: str | Path | None = None,
) -> Any:
    """Train TfidfVectorizer + LogisticRegression classifier with strict split check."""
    # Anti-leakage guard: must be in train split
    with GuardedSplitContext(split):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import joblib

        texts = [clean_contract_text(t) for t, _ in train_data]
        labels = [lbl for _, lbl in train_data]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])

        pipeline.fit(texts, labels)

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, out_file)

        return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ContractIQ clause classifier.")
    parser.add_argument("--cuad-path", type=str, default=None, help="Path to CUAD v1 JSON file")
    parser.add_argument("--output", type=str, default="ml/results/tfidf_classifier.joblib")
    args = parser.parse_args()

    if args.cuad_path and Path(args.cuad_path).exists():
        contracts = load_cuad(args.cuad_path)
        splits = make_splits(contracts)
        train_contracts = get_split_contracts(contracts, splits, "train")
        train_examples = extract_training_examples(train_contracts)
        if not train_examples:
            train_examples = SAMPLE_TRAIN_CORPUS
    else:
        train_examples = SAMPLE_TRAIN_CORPUS

    pipeline = train_classifier(train_examples, split="train", output_path=args.output)
    print(f"Classifier trained on {len(train_examples)} examples and saved to {args.output}")


if __name__ == "__main__":
    main()
