"""Utilities for extracting text and structure hints from PDF files.

The reader exposes both plain-text extraction and layout-aware extraction. The
layout-aware form preserves spacing and line breaks more faithfully, which makes
it more useful for tables, section boundaries, and heading detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from pypdf import PdfReader


def resolve_pdf_path(pdf_path: str | Path) -> Path:
    """Resolve a PDF path and validate that it exists."""
    path = Path(pdf_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    return path


def _normalize_page_numbers(page_numbers: Sequence[int] | None, page_count: int) -> list[int]:
    if not page_numbers:
        return list(range(1, page_count + 1))

    normalized: list[int] = []
    for page_number in page_numbers:
        if page_number < 1 or page_number > page_count:
            raise ValueError(
                f"Page {page_number} is out of range for a {page_count}-page PDF"
            )
        if page_number not in normalized:
            normalized.append(page_number)
    return normalized


def _trim_text(text: str, max_chars: int | None) -> tuple[str, bool, int]:
    original_length = len(text)
    if max_chars is None or max_chars <= 0 or original_length <= max_chars:
        return text, False, original_length
    return text[:max_chars], True, original_length


def _clean_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _clean_metadata(metadata: Any) -> dict[str, Any]:
    if not metadata:
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        cleaned[str(key).lstrip("/")] = _clean_metadata_value(value)
    return cleaned


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False

    words = stripped.split()
    if len(words) > 12:
        return False

    if re.match(r"^\d+(?:\.\d+)*[\):.-]?\s+\S+", stripped):
        return True
    if stripped.endswith(":"):
        return True

    letters = re.sub(r"[^A-Za-z]", "", stripped)
    if letters and letters.upper() == letters and len(words) <= 8:
        return True

    title_word_count = sum(1 for word in words if word[:1].isupper())
    if title_word_count >= max(1, len(words) - 1) and len(words) <= 8:
        return True

    return False


def detect_heading_candidates(text: str, max_candidates: int = 12) -> list[str]:
    """Return likely section headings from layout-oriented PDF text."""
    seen: set[str] = set()
    headings: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not _looks_like_heading(stripped):
            continue
        if stripped in seen:
            continue

        seen.add(stripped)
        headings.append(stripped)
        if len(headings) >= max_candidates:
            break

    return headings


def detect_table_blocks(text: str, max_blocks: int = 6) -> list[dict[str, Any]]:
    """Return contiguous layout lines that look tabular."""
    blocks: list[dict[str, Any]] = []
    current_lines: list[tuple[int, str]] = []

    def flush() -> None:
        if len(current_lines) < 2:
            current_lines.clear()
            return

        start_line = current_lines[0][0]
        end_line = current_lines[-1][0]
        raw_lines = [line for _, line in current_lines]
        blocks.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "row_count": len(raw_lines),
                "preview": "\n".join(raw_lines[:8]),
            }
        )
        current_lines.clear()

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.rstrip()
        looks_tabular = bool(re.search(r"\S\s{2,}\S", stripped) or "\t" in stripped)
        if stripped and looks_tabular:
            current_lines.append((line_number, stripped))
            continue

        flush()
        if len(blocks) >= max_blocks:
            break

    flush()
    return blocks[:max_blocks]


def _flatten_outline_items(
    reader: PdfReader,
    outline_items: Iterable[Any],
    depth: int,
    output: list[dict[str, Any]],
) -> None:
    for item in outline_items:
        if isinstance(item, list):
            _flatten_outline_items(reader, item, depth + 1, output)
            continue

        title = getattr(item, "title", None)
        if not title:
            continue

        page_number: int | None = None
        try:
            page_number = reader.get_destination_page_number(item) + 1
        except Exception:
            page_number = None

        output.append(
            {
                "title": str(title),
                "depth": depth,
                "page_number": page_number,
            }
        )


def extract_outline(reader: PdfReader) -> list[dict[str, Any]]:
    """Extract the bookmark/outline tree into a flat, JSON-safe form."""
    outline = getattr(reader, "outline", None)
    if not outline:
        return []

    flattened: list[dict[str, Any]] = []
    if isinstance(outline, list):
        _flatten_outline_items(reader, outline, 0, flattened)
    return flattened


def read_pdf(
    pdf_path: str | Path,
    page_numbers: Sequence[int] | None = None,
    *,
    max_chars_per_page: int | None = 8000,
    include_plain_text: bool = True,
    include_layout_text: bool = True,
    include_outline: bool = True,
) -> dict[str, Any]:
    """Read a PDF and return text plus structure-oriented hints.

    The returned payload is JSON-serializable so it can be used directly from
    CLI tooling or MCP tools.
    """
    path = resolve_pdf_path(pdf_path)
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    selected_pages = _normalize_page_numbers(page_numbers, page_count)

    result: dict[str, Any] = {
        "pdf_path": str(path),
        "page_count": page_count,
        "selected_pages": selected_pages,
        "metadata": _clean_metadata(reader.metadata),
        "outline": extract_outline(reader) if include_outline else [],
        "pages": [],
    }

    for page_number in selected_pages:
        page = reader.pages[page_number - 1]
        plain_text = page.extract_text(extraction_mode="plain") or ""
        layout_text = page.extract_text(extraction_mode="layout") or ""

        plain_payload, plain_truncated, plain_length = _trim_text(
            plain_text, max_chars_per_page if include_plain_text else 0
        )
        layout_payload, layout_truncated, layout_length = _trim_text(
            layout_text, max_chars_per_page if include_layout_text else 0
        )

        structure_source = layout_text or plain_text
        page_payload = {
            "page_number": page_number,
            "plain_text": plain_payload if include_plain_text else "",
            "layout_text": layout_payload if include_layout_text else "",
            "plain_text_length": plain_length,
            "layout_text_length": layout_length,
            "plain_text_truncated": plain_truncated if include_plain_text else False,
            "layout_text_truncated": layout_truncated if include_layout_text else False,
            "heading_candidates": detect_heading_candidates(structure_source),
            "table_blocks": detect_table_blocks(structure_source),
        }
        result["pages"].append(page_payload)

    return result


def extract_pdf_page(
    pdf_path: str | Path,
    page_number: int,
    *,
    max_chars: int | None = 8000,
) -> dict[str, Any]:
    """Convenience wrapper for reading a single page."""
    result = read_pdf(
        pdf_path,
        [page_number],
        max_chars_per_page=max_chars,
        include_plain_text=True,
        include_layout_text=True,
        include_outline=False,
    )
    result["page"] = result["pages"][0]
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract plain text and layout-aware structure hints from a PDF."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        help="Optional one-based page numbers to extract",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum characters to keep per page for each text mode",
    )
    parser.add_argument(
        "--no-plain-text",
        action="store_true",
        help="Skip the standard text extraction payload",
    )
    parser.add_argument(
        "--no-layout-text",
        action="store_true",
        help="Skip the layout-aware text extraction payload",
    )
    parser.add_argument(
        "--no-outline",
        action="store_true",
        help="Skip bookmark/outline extraction",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = read_pdf(
        args.pdf_path,
        args.pages,
        max_chars_per_page=args.max_chars,
        include_plain_text=not args.no_plain_text,
        include_layout_text=not args.no_layout_text,
        include_outline=not args.no_outline,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()