"""MCP server exposing prompt quality validation tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from prompt_validator import validate_prompt


mcp = FastMCP(
    name="prompt-tools",
    instructions=(
        "Use this server when you need to review, validate, or improve a user prompt "
        "before sending it to a model. Prefer the structured validator output when you "
        "need a score, missing-context checks, or a suggested rewrite."
    ),
)


@mcp.tool(
    description=(
        "Validate the quality of a user prompt and return a score, strengths, issues, "
        "follow-up questions, and a suggested rewrite."
    ),
    structured_output=True,
    meta={"anthropic/maxResultSizeChars": 40000},
)
def validate_user_prompt(prompt: str, task_type: str | None = None) -> dict[str, Any]:
    """Return a structured prompt quality assessment."""
    return validate_prompt(prompt, task_type=task_type)


if __name__ == "__main__":
    mcp.run("stdio")