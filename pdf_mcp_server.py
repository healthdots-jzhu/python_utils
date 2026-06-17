"""MCP server exposing layout-aware PDF inspection tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from pdf_reader import extract_pdf_page, read_pdf
from pdf_writer import write_pdf


mcp = FastMCP(
    name="pdf-tools",
    instructions=(
        "Use this server when you need to inspect existing PDFs or generate new "
        "PDFs from source files. Prefer the layout-aware output when tables, "
        "section boundaries, or heading hierarchy matter."
    ),
)


@mcp.tool(
    description=(
        "Read a PDF and return metadata, outline, plain text, layout-aware text, "
        "heading candidates, and likely table blocks."
    ),
    structured_output=True,
    meta={"anthropic/maxResultSizeChars": 200000},
)
def inspect_pdf(
    pdf_path: str,
    page_numbers: list[int] | None = None,
    max_chars_per_page: int = 8000,
    include_plain_text: bool = True,
    include_layout_text: bool = True,
    include_outline: bool = True,
) -> dict[str, Any]:
    """Return a layout-aware PDF inspection payload."""
    return read_pdf(
        pdf_path,
        page_numbers,
        max_chars_per_page=max_chars_per_page,
        include_plain_text=include_plain_text,
        include_layout_text=include_layout_text,
        include_outline=include_outline,
    )


@mcp.tool(
    description=(
        "Read a single PDF page and return both plain and layout-aware text plus "
        "detected headings and likely table blocks."
    ),
    structured_output=True,
    meta={"anthropic/maxResultSizeChars": 120000},
)
def inspect_pdf_page(
    pdf_path: str,
    page_number: int,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Return a focused inspection payload for one page."""
    return extract_pdf_page(pdf_path, page_number, max_chars=max_chars)


@mcp.tool(
    description=(
        "Write a PDF from text, Markdown, XML, YAML, HTML, CSV, Excel, or Word "
        "source files and return the generated output path and page count."
    ),
    structured_output=True,
    meta={"anthropic/maxResultSizeChars": 40000},
)
def write_pdf_from_file(
    source_path: str,
    output_path: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create a PDF from a supported source document."""
    return write_pdf(source_path, output_path, title=title)


if __name__ == "__main__":
    mcp.run("stdio")