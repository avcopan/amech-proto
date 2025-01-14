"""Column schema."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def orig(col_: str | Sequence[str]) -> str | list[str]:
    """Add `orig_` prefix to column name.

    :param col: Column name
    :return: Column name
    """
    fmt_ = "orig_{}".format
    return fmt_(col_) if isinstance(col_, str) else list(map(fmt_, col_))


def to_orig(col_: str | Sequence[str]) -> dict[str, str]:
    """Create mapping to `orig_` columns.

    :param cols: Column names
    :return: Mapping
    """
    col_ = [col_] if isinstance(col_, str) else col_
    return {c: orig(c) for c in col_}


def from_orig(col_: str | Sequence[str]) -> dict[str, str]:
    """Create mapping from `orig_` columns.

    :param cols: Column names
    :return: Mapping
    """
    col_ = [col_] if isinstance(col_, str) else col_
    return {orig(c): c for c in col_}
