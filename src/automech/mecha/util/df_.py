"""DataFrame utilities
"""

from typing import Dict, Optional, Tuple, Union

import pandas

Key = str
Keys = Tuple[str, ...]
Key_ = Union[Key, Keys]
Value = object
Values = Tuple[object, ...]
Value_ = Union[Value, Values]


def from_csv(path: str) -> pandas.DataFrame:
    """Read a dataframe from a CSV file

    :param path: The path to the CSV file
    :return: The dataframe
    """
    try:
        df = pandas.read_csv(path)
    except pandas.errors.ParserError:
        df = pandas.read_csv(path, quotechar="'")
    return df


def to_csv(df: pandas.DataFrame, path: Optional[str]):
    """Write a dataframe to a CSV file

    If `path` is `None`, this function does nothing

    :param path: The path to the CSV file
    """
    if path is not None:
        df.to_csv(path, index=False)


def lookup_dict(
    df: pandas.DataFrame, in_: Optional[Key_] = None, out_: Optional[Key_] = None
) -> Dict[Value_, Value_]:
    """Form a lookup dictionary mapping one column onto another in a dataframe

    Allows mappings between sets of columns, using a tuple of column keys

    :param df: The dataframe
    :param in_: The input key or keys; if `None` the index is used
    :param out_: The output key or keys; if `None` the index is used
    :return: The dictionary mapping input values to output values
    """

    def check_(key_):
        return (
            True
            if key_ is None
            else key_ in df if isinstance(key_, str) else all(k in df for k in key_)
        )

    def values_(key_):
        return (
            df.index
            if key_ is None
            else df[key_] if isinstance(key_, str) else zip(*(df[k] for k in key_))
        )

    assert check_(in_), f"{in_} not in {df}"
    assert check_(out_), f"{out_} not in {df}"

    return dict(zip(values_(in_), values_(out_)))
