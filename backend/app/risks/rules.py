"""Deterministic YAML-backed risk assessment for clause predictions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.clauses.models import ClausePrediction


@dataclass(frozen=True)
class RiskRule:
    id: str
    clause_type: str
    risk_level: str
    patterns: tuple[str, ...]
    explanation: str


def load_risk_rules(path: str | Path | None = None) -> tuple[RiskRule, ...]:
    """Load and validate the bundled deterministic rules."""
    rules_path = Path(path) if path is not None else Path(__file__).with_name("rules.yaml")
    with rules_path.open(encoding="utf-8") as rules_file:
        data = yaml.safe_load(rules_file) or {}

    rules: list[RiskRule] = []
    for raw_rule in data.get("rules", []):
        required = {"id", "clause_type", "risk_level", "patterns", "explanation"}
        if not isinstance(raw_rule, dict) or not required <= raw_rule.keys():
            raise ValueError("Each risk rule must define id, clause_type, risk_level, patterns, and explanation")
        if raw_rule["risk_level"] not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"Unsupported risk level: {raw_rule['risk_level']!r}")
        if not isinstance(raw_rule["patterns"], list) or not raw_rule["patterns"]:
            raise ValueError(f"Risk rule {raw_rule['id']!r} must define at least one pattern")
        rules.append(
            RiskRule(
                id=str(raw_rule["id"]),
                clause_type=str(raw_rule["clause_type"]),
                risk_level=str(raw_rule["risk_level"]),
                patterns=tuple(str(pattern) for pattern in raw_rule["patterns"]),
                explanation=str(raw_rule["explanation"]),
            )
        )
    return tuple(rules)


def assess_clause_prediction(
    prediction: ClausePrediction, rules: tuple[RiskRule, ...] | None = None
) -> list[RiskRule]:
    """Return rules matched against one already-classified clause prediction."""
    active_rules = load_risk_rules() if rules is None else rules
    return [
        rule
        for rule in active_rules
        if rule.clause_type == prediction.clause_type
        and any(re.search(pattern, prediction.span_text, re.IGNORECASE) for pattern in rule.patterns)
    ]
