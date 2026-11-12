"""
MobileLLM CLI package.
"""

from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    from mobilellm.cli.main import main as _main

    return _main(argv)


__all__ = ["main"]
