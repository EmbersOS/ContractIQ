"""Rule-based candidate clause scanner — no ML, pure Python regex."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.contracts.extractor import PageOffset


@dataclass
class CandidateClause:
    clause_type: str
    span_text: str
    char_start: int
    char_end: int
    matched_keyword: str
    page_num: int = 1


# Each entry: (clause_type, list_of_regex_patterns)
# Patterns use re.IGNORECASE
_CLAUSE_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "Limitation of Liability",
        [
            r"limitation[s]? of liability",
            r"limit(?:ed)? liability",
            r"in no event shall .{0,80}liable",
            r"aggregate liability .{0,50}shall not exceed",
            r"maximum liability",
            r"cap on liability",
        ],
    ),
    (
        "Indemnification",
        [
            r"indemnif(?:y|ication|ied|ies)",
            r"hold harmless",
            r"defend .{0,60}from .{0,60}claims",
        ],
    ),
    (
        "Termination",
        [
            r"terminat(?:e|ion|ed|ing)\b",
            r"right to terminate",
            r"notice of termination",
            r"upon termination",
        ],
    ),
    (
        "Governing Law",
        [
            r"governed by (?:the )?laws? of",
            r"governing law",
            r"jurisdiction of .{0,60}courts?",
            r"laws? of the (?:state|country|province) of",
        ],
    ),
    (
        "Confidentiality",
        [
            r"confidential(?:ity|information)?",
            r"non-disclosure",
            r"nda\b",
            r"proprietary information",
            r"trade secret",
        ],
    ),
    (
        "Non-Compete",
        [
            r"non-?compete",
            r"not (?:to )?compete",
            r"covenant not to compete",
            r"restrictive covenant",
            r"non-?solicitation",
        ],
    ),
    (
        "Warranty",
        [
            r"warrant(?:y|ies|s|ed)\b",
            r"as[- ]is\b",
            r"no warranty",
            r"disclaimer of warrant(?:y|ies)",
            r"merchantabilit(?:y|ies)",
            r"fitness for a particular purpose",
        ],
    ),
    (
        "Intellectual Property",
        [
            r"intellectual property",
            r"\bip\b.{0,30}rights?",
            r"patent(?:s|ed)?\b",
            r"copyright\b",
            r"trademark\b",
            r"work[s]? made for hire",
            r"assignment of .{0,40}rights?",
        ],
    ),
    (
        "Payment Terms",
        [
            r"payment terms?",
            r"due (?:within|in) \d+ days?",
            r"invoice(?:d|s)?\b",
            r"net[- ]\d+",
            r"late (?:payment|fee)",
            r"interest on (?:overdue|late|unpaid)",
            r"billing cycle",
        ],
    ),
    (
        "Dispute Resolution",
        [
            r"dispute resolution",
            r"arbitrat(?:e|ion|or)",
            r"mediat(?:e|ion)",
            r"escalation procedure",
            r"good[ -]faith (?:negotiation|effort)",
        ],
    ),
]

# Compile all patterns once at module load
_COMPILED: list[tuple[str, list[tuple[str, re.Pattern[str]]]]] = [
    (clause_type, [(pat, re.compile(pat, re.IGNORECASE)) for pat in patterns])
    for clause_type, patterns in _CLAUSE_PATTERNS
]


def _find_page_number(char_pos: int, page_offsets: list[PageOffset] | None) -> int:
    if not page_offsets:
        return 1
    for po in page_offsets:
        if po.char_start <= char_pos < po.char_end:
            return po.page_num
    return page_offsets[-1].page_num if page_offsets else 1


def find_clause_candidates(
    text: str, page_offsets: list[PageOffset] | None = None
) -> list[CandidateClause]:
    """Scan *text* and return a list of CandidateClause matches.

    Uses pure regex — no ML. Each match records the span text inline
    so the source text never needs to be re-fetched.
    """
    results: list[CandidateClause] = []
    seen_spans: set[tuple[int, int]] = set()

    for clause_type, pattern_pairs in _COMPILED:
        for raw_pattern, compiled in pattern_pairs:
            for match in compiled.finditer(text):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                page = _find_page_number(match.start(), page_offsets)
                results.append(
                    CandidateClause(
                        clause_type=clause_type,
                        span_text=match.group(0),
                        char_start=match.start(),
                        char_end=match.end(),
                        matched_keyword=raw_pattern,
                        page_num=page,
                    )
                )

    results.sort(key=lambda c: c.char_start)
    return results


def candidates_to_dict(candidates: list[CandidateClause]) -> list[dict]:
    """Serialize CandidateClause list to plain dicts for JSON storage."""
    return [
        {
            "clause_type": c.clause_type,
            "span_text": c.span_text,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "matched_keyword": c.matched_keyword,
            "page_num": c.page_num,
        }
        for c in candidates
    ]

