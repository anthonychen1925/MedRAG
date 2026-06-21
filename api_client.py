"""Anthropic client wrapper for MedRAG.

Loads the CONTEXT.md system prompt and calls Claude Opus 4.8 to produce the
structured safety report. The model's clinical behavior is defined entirely by
CONTEXT.md (the system prompt) plus the assembled user prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import settings

CONTEXT_PATH = Path(__file__).parent / "CONTEXT.md"


@lru_cache(maxsize=1)
def load_system_context() -> str:
    """Read CONTEXT.md (the system prompt). Cached after first read."""
    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(f"CONTEXT.md not found at {CONTEXT_PATH}")
    return CONTEXT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _get_client():
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError("anthropic is not installed. Run `pip install anthropic`.") from exc

    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env to generate reports."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def generate_report(
    prompt: str,
    system_context: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Send the assembled prompt to Claude and return the report text."""
    client = _get_client()
    system = system_context if system_context is not None else load_system_context()

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
