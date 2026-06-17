"""Prompt quality validation helpers for chat and agent workflows."""

from __future__ import annotations

import re
from typing import Any


CONTEXT_KEYWORDS = {
    "attached",
    "audience",
    "context",
    "dataset",
    "example",
    "examples",
    "file",
    "files",
    "input",
    "inputs",
    "language",
    "repo",
    "repository",
    "source",
    "using",
}

OUTPUT_KEYWORDS = {
    "answer",
    "bullet",
    "bullets",
    "code",
    "csv",
    "explain",
    "format",
    "json",
    "list",
    "output",
    "report",
    "response",
    "return",
    "rewrite",
    "step",
    "steps",
    "summary",
    "table",
}

CONSTRAINT_KEYWORDS = {
    "at most",
    "avoid",
    "exactly",
    "must",
    "no more than",
    "only",
    "without",
    "should",
    "do not",
    "don't",
    "under",
}

AMBIGUOUS_REFERENCES = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "thing",
    "stuff",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text.lower())


def _contains_any_keyword(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _extract_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text) if segment.strip()]
    return sentences


def _make_suggested_rewrite(prompt: str, issues: list[str], task_type: str | None) -> str:
    normalized = _normalize_whitespace(prompt)
    if not normalized:
        return "Goal: [what you want]\nContext: [relevant inputs or files]\nOutput: [required format]\nConstraints: [limits, style, must/avoid]"

    rewrite_lines = [f"Goal: {normalized}"]
    if any("context" in issue.lower() for issue in issues):
        rewrite_lines.append("Context: Include the relevant files, data, audience, or examples.")
    if any("output" in issue.lower() or "format" in issue.lower() for issue in issues):
        rewrite_lines.append("Output: Specify the format you want back, for example bullets, JSON, code, or a short summary.")
    if any("constraint" in issue.lower() or "limit" in issue.lower() for issue in issues):
        rewrite_lines.append("Constraints: State any must-have rules, length limits, exclusions, or style requirements.")
    if task_type:
        rewrite_lines.append(f"Task type: {task_type}")
    return "\n".join(rewrite_lines)


def validate_prompt(prompt: str, task_type: str | None = None) -> dict[str, Any]:
    """Return a structured prompt quality assessment."""
    normalized = _normalize_whitespace(prompt)
    words = _tokenize_words(normalized)
    sentences = _extract_sentences(prompt)

    issues: list[str] = []
    strengths: list[str] = []
    suggested_questions: list[str] = []
    score = 100

    if not normalized:
        issues.append("Prompt is empty.")
        score -= 90
    elif len(words) < 8:
        issues.append("Prompt is very short and likely underspecified.")
        suggested_questions.append("What exact task should be completed?")
        score -= 30
    else:
        strengths.append("Prompt states a concrete task.")

    has_context = _contains_any_keyword(normalized, CONTEXT_KEYWORDS)
    if has_context:
        strengths.append("Prompt includes contextual cues or source references.")
    else:
        issues.append("Prompt is missing concrete context such as files, inputs, audience, or examples.")
        suggested_questions.append("What context, files, or examples should the model use?")
        score -= 15

    has_output = _contains_any_keyword(normalized, OUTPUT_KEYWORDS)
    if has_output:
        strengths.append("Prompt gives some hint about the desired output.")
    else:
        issues.append("Prompt does not clearly specify the desired output format.")
        suggested_questions.append("What should the final output look like?")
        score -= 15

    has_constraints = _contains_any_keyword(normalized, CONSTRAINT_KEYWORDS)
    if has_constraints:
        strengths.append("Prompt includes constraints or preferences.")
    else:
        issues.append("Prompt has no explicit constraints, limits, or must/avoid rules.")
        suggested_questions.append("Are there any constraints on length, tone, format, or scope?")
        score -= 10

    ambiguous_count = sum(1 for word in words if word in AMBIGUOUS_REFERENCES)
    if ambiguous_count >= 3 and not has_context:
        issues.append("Prompt uses ambiguous references without enough grounding context.")
        suggested_questions.append("What do words like 'this' or 'it' refer to exactly?")
        score -= 10

    if len(sentences) > 6 and not has_output:
        issues.append("Prompt is detailed but still lacks a clear success condition or output shape.")
        score -= 5

    if task_type:
        strengths.append(f"Prompt is being reviewed for task type: {task_type}.")

    score = max(0, min(score, 100))
    blocking_issues = [issue for issue in issues if "empty" in issue.lower() or "underspecified" in issue.lower()]
    ready_for_model = score >= 70 and not blocking_issues

    return {
        "score": score,
        "ready_for_model": ready_for_model,
        "task_type": task_type or "general",
        "word_count": len(words),
        "strengths": strengths,
        "issues": issues,
        "blocking_issues": blocking_issues,
        "suggested_questions": suggested_questions,
        "suggested_rewrite": _make_suggested_rewrite(prompt, issues, task_type),
    }