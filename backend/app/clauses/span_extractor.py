"""Deterministic extraction of clause bodies from structured contract text."""
from __future__ import annotations

import re


_CLAUSE_HEADINGS = {
    "Limitation of Liability": r"limitation\s+of\s+liability",
    "Indemnification": r"indemnification",
    "Termination": r"termination",
    "Governing Law": r"governing\s+law",
    "Confidentiality": r"confidentiality",
    "Non-Compete": r"non[-\s]?compete",
    "Warranty": r"warrant(?:y|ies)",
    "Intellectual Property": r"intellectual\s+property",
    "Payment Terms": r"payment\s+terms?",
    "Dispute Resolution": r"dispute\s+resolution",
}

_NUMBERED_ALIASES = {
    "Payment Terms": r"payment",
    "Dispute Resolution": r"dispute",
}


def _compile_heading(clause_pattern: str, numbered_alias: str | None = None) -> re.Pattern[str]:
    """Compile a line-anchored heading, rather than matching prose mentions."""
    full_heading = rf"""(?im)
        ^[ \t]*
        (?:\d+(?:\.\d+)*[.)\-:]?[ \t]+)?
        (?:{clause_pattern})
        [ \t]*(?:[.:\-][ \t]*|\r?\n|$)
    """
    if numbered_alias is None:
        return re.compile(full_heading, re.VERBOSE)

    # CUAD's PAYMENT and DISPUTE headings are abbreviated, so accept them only
    # when their numeric section prefix makes them structurally unambiguous.
    return re.compile(
        rf"""{full_heading}|
        ^[ \t]*
        \d+(?:\.\d+)*[.)\-:]?[ \t]+
        (?:{numbered_alias})
        [ \t]*(?:[.:\-][ \t]*|\r?\n|$)
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )


_COMPILED_HEADINGS = {
    clause_type: _compile_heading(pattern, _NUMBERED_ALIASES.get(clause_type))
    for clause_type, pattern in _CLAUSE_HEADINGS.items()
}

# A numbered heading, or an unnumbered all-caps/title-case heading, is a credible
# boundary. Requiring a delimiter prevents ordinary sentences from ending a span.
_SECTION_HEADING_RE = re.compile(
    r"""(?im)
    ^[ \t]*
    (?:
        \d+(?:\.\d+)*[.)\-:]?[ \t]+
        [^\r\n]{1,100}?
      |
        (?:[A-Z][A-Z &/,'\-]{1,99}|[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,9})
    )
    [ \t]*(?:[.:\-][ \t]*|\r?\n|$)
    """,
    re.VERBOSE,
)


def extract_span_for_clause(source_text: str, predicted_clause_type: str) -> tuple[str, int, int]:
    """Return a predicted clause body and its raw-source character offsets.

    The classifier supplies *predicted_clause_type*; this function only locates
    the corresponding structured section. It never normalizes *source_text*, so
    returned offsets remain valid for CUAD and extracted contract text.
    """
    heading = _COMPILED_HEADINGS.get(predicted_clause_type)
    if not source_text or heading is None:
        return "", 0, 0

    match = heading.search(source_text)
    if match is None:
        return "", 0, 0

    start = match.end()
    while start < len(source_text) and source_text[start].isspace():
        start += 1

    next_heading = _SECTION_HEADING_RE.search(source_text, start)
    end = next_heading.start() if next_heading else len(source_text)
    while end > start and source_text[end - 1].isspace():
        end -= 1

    if start >= end:
        return "", 0, 0
    return source_text[start:end], start, end
