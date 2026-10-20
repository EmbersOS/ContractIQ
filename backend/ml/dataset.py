"""CUAD dataset loading and deterministic contract-level split generation."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_cuad(cuad_path: str | Path) -> list[dict[str, Any]]:
    """Load CUAD dataset in SQuAD v2 JSON format.

    Returns a flat list of contract dictionaries with:
    - contract_id: unique identifier derived from title or index
    - title: contract title
    - context: full contract text
    - qas: list of question-answer pairs with clause type and character spans
    """
    path = Path(cuad_path)
    if not path.exists():
        raise FileNotFoundError(f"CUAD dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contracts: list[dict[str, Any]] = []
    items = data.get("data", []) if isinstance(data, dict) else data

    for idx, item in enumerate(items):
        title = item.get("title", f"contract_{idx}")
        paragraphs = item.get("paragraphs", [])
        for p_idx, para in enumerate(paragraphs):
            context = para.get("context", "")
            qas = para.get("qas", [])
            contract_id = f"{title}__{p_idx}" if len(paragraphs) > 1 else title
            contracts.append(
                {
                    "contract_id": contract_id,
                    "title": title,
                    "context": context,
                    "qas": qas,
                }
            )

    return contracts


def make_splits(
    cuad_source: str | Path | list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Generate deterministic, contract-level splits for CUAD.

    All question-answer pairs from the same contract are strictly confined to
    one split (train, val, or test) to prevent cross-split data leakage.
    """
    if isinstance(cuad_source, (str, Path)):
        contracts = load_cuad(cuad_source)
    else:
        contracts = cuad_source

    all_ids = sorted({c["contract_id"] for c in contracts})
    if not all_ids:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    shuffled = list(all_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Ensure at least 1 per split when n >= 3
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])

    # Strict anti-leakage invariants
    assert train_ids & val_ids == set(), "Leakage detected: train & val overlap"
    assert train_ids & test_ids == set(), "Leakage detected: train & test overlap"
    assert val_ids & test_ids == set(), "Leakage detected: val & test overlap"

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def verify_no_overlap(splits: dict[str, list[str]]) -> bool:
    """Verify that split sets contain zero overlapping contract IDs."""
    train = set(splits.get("train", []))
    val = set(splits.get("val", []))
    test = set(splits.get("test", []))

    if train & val or train & test or val & test:
        return False
    return True


def get_split_contracts(
    contracts: list[dict[str, Any]],
    splits: dict[str, list[str]],
    split_name: str,
) -> list[dict[str, Any]]:
    """Filter loaded contracts for the given split ('train', 'val', or 'test')."""
    target_ids = set(splits.get(split_name, []))
    return [c for c in contracts if c["contract_id"] in target_ids]
