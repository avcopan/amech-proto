"""Column schema."""

from collections.abc import Sequence


def orig(col: str) -> str:
    """Add `orig_` prefix to column name.

    :param col: Column name
    :return: Column name
    """
    return f"orig_{col}"


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
