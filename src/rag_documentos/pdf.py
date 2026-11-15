"""PDF text extraction, page by page."""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class PDFError(ValueError):
    """The file is not a readable PDF."""


class NoTextError(PDFError):
    """The PDF has no extractable text (for example, scanned images without OCR)."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as shown in PDF viewers
    text: str


@dataclass(frozen=True)
class ParsedPDF:
    pages: list[Page]  # only pages that contain text
    total_pages: int
    empty_pages: list[int]


_SOFT_HYPHEN = re.compile(r"(?<=\w)-\n(?=[a-zà-öø-ÿ])")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Normalise text extracted from a PDF page.

    Joins words hyphenated across line breaks, collapses line wraps into spaces and
    keeps blank lines as paragraph separators.
    """
    text = unicodedata.normalize("NFKC", raw.replace("\x00", ""))
    text = _SOFT_HYPHEN.sub("", text)
    paragraphs = (_WHITESPACE.sub(" ", p).strip() for p in _PARAGRAPH_BREAK.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def extract_pages(source: bytes | str | Path) -> ParsedPDF:
    """Extract the text of every page.

    Raises:
        PDFError: the file cannot be parsed or is password protected.
        NoTextError: no page has extractable text.
    """
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PDFError("O PDF está protegido por senha.")
        total = len(reader.pages)
        raw_pages = []
        for page in reader.pages:
            try:
                raw_pages.append(page.extract_text() or "")
            except Exception:  # a single broken page should not sink the document
                raw_pages.append("")
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError(f"Não foi possível ler o PDF: {exc}") from exc

    pages: list[Page] = []
    empty: list[int] = []
    for number, raw in enumerate(raw_pages, start=1):
        text = clean_text(raw)
        if text:
            pages.append(Page(number=number, text=text))
        else:
            empty.append(number)

    if not pages:
        raise NoTextError(
            "Nenhum texto extraível foi encontrado. Se o PDF é escaneado, "
            "aplique OCR antes de enviá-lo."
        )
    return ParsedPDF(pages=pages, total_pages=total, empty_pages=empty)
