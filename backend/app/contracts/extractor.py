"""Text extraction from PDF and DOCX files."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageOffset:
    page_num: int
    char_start: int
    char_end: int


@dataclass
class ExtractionResult:
    text: str
    page_offsets: list[PageOffset] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


def extract_pdf(path: str) -> ExtractionResult:
    """Extract text from a PDF file using pdfplumber.

    Returns an ExtractionResult with full concatenated text and per-page offsets.
    """
    import pdfplumber  # type: ignore

    parts: list[str] = []
    offsets: list[PageOffset] = []
    cumulative = 0

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text: str = page.extract_text() or ""
            # Add a newline separator between pages
            if parts:
                page_text = "\n" + page_text
            char_start = cumulative
            char_end = cumulative + len(page_text)
            offsets.append(PageOffset(page_num=i + 1, char_start=char_start, char_end=char_end))
            parts.append(page_text)
            cumulative = char_end

    full_text = "".join(parts)
    return ExtractionResult(text=full_text, page_offsets=offsets)


def extract_docx(path: str) -> ExtractionResult:
    """Extract text from a DOCX file using python-docx.

    Returns an ExtractionResult with full text and per-paragraph offsets
    (treated as 'pages' for schema compatibility).
    """
    from docx import Document  # type: ignore

    doc = Document(path)
    parts: list[str] = []
    offsets: list[PageOffset] = []
    cumulative = 0

    for i, para in enumerate(doc.paragraphs):
        para_text = para.text
        if parts:
            para_text = "\n" + para_text
        char_start = cumulative
        char_end = cumulative + len(para_text)
        offsets.append(PageOffset(page_num=i + 1, char_start=char_start, char_end=char_end))
        parts.append(para_text)
        cumulative = char_end

    full_text = "".join(parts)
    return ExtractionResult(text=full_text, page_offsets=offsets)

