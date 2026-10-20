"""Persistence service for deterministic risk findings."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.clauses.models import ClausePrediction
from app.risks.models import RiskFinding
from app.risks.rules import RiskRule, assess_clause_prediction


def persist_risk_findings(
    db: Session, prediction: ClausePrediction, rules: tuple[RiskRule, ...] | None = None
) -> list[RiskFinding]:
    """Persist each matched rule once for a clause prediction."""
    matched_rules = assess_clause_prediction(prediction, rules)
    if not matched_rules:
        return []

    existing_rule_ids = {
        rule_id
        for (rule_id,) in db.query(RiskFinding.rule_id).filter(RiskFinding.clause_id == prediction.id).all()
    }
    findings = [
        RiskFinding(
            contract_id=prediction.contract_id,
            clause_id=prediction.id,
            rule_id=rule.id,
            risk_level=rule.risk_level,
            explanation=rule.explanation,
            span_text=prediction.span_text,
            char_start=prediction.char_start,
            char_end=prediction.char_end,
        )
        for rule in matched_rules
        if rule.id not in existing_rule_ids
    ]
    if findings:
        db.add_all(findings)
        db.commit()
        for finding in findings:
            db.refresh(finding)
    return findings
