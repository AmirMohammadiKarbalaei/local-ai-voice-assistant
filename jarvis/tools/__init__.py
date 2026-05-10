"""Tool package for Jarvis.

This package replaces the old single `jarvis/tools.py` module.

Important:
    Do not keep both:
        jarvis/tools.py
    and:
        jarvis/tools/

Rename or delete the old `tools.py` first.
"""

from .registry import ToolRegistry

__all__ = ["ToolRegistry"]
