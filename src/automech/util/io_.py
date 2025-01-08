"""Convenience functions for I/O."""

import os
from pathlib import Path


def read_text(inp: str | Path) -> str:
    """Read text input as a string.

    :param inp: Text input Path object, path string, or contents string.
    :return: Text
    """
    return (
        Path(inp).read_text()
        if isinstance(inp, Path) or os.path.exists(inp)
        else str(inp)
    )
