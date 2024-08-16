"""DataFrame utilities."""

from collections.abc import Callable, Sequence
from pathlib import Path

import narwhals
import polars
from narwhals.typing import FrameT
from tqdm.auto import tqdm

Key = str
Keys = Sequence[str]
Key_ = Key | Keys
Value = object
Values = Sequence[object]
Value_ = Value | Values


def from_csv(path: str) -> polars.DataFrame:
    """Read a dataframe from a CSV file.

    :param path: The path to the CSV file
    :return: The dataframe
    """
    try:
        df = polars.read_csv(path)
    except polars.exceptions.ComputeError:
        df = polars.read_csv(path, quote_char="'")
    return df


def to_csv(
    df: polars.DataFrame, path: str | None, quote_char: str | None = None
) -> None:
    """Write a dataframe to a CSV file.

    If `path` is `None`, this function does nothing

    :param df: The dataframe
    :param path: The path to the CSV file
    :param quote_char: Optionally, override the default quote character
    """
    kwargs = {} if quote_char is None else {"quote_char": quote_char}
    if path is not None:
        path: Path = Path(path)
        path = path if path.suffix == ".csv" else path.with_suffix(".csv")
        df.write_csv(path, **kwargs)


@narwhals.narwhalify
def map_(
    df: FrameT,
    in_: Key_,
    out: Key | None,
    func_: Callable,
    dct: dict[object, object] | None = None,
    dtype: polars.DataType | None = None,
    bar: bool = True,
) -> FrameT:
    """Map columns from a dataframe onto a new column.

    :param df: The dataframe
    :param in_: The input key or keys
    :param out: The output key; if `None`, the function output will be ignored
    :param func_: The mapping function
    :param dct: A lookup dictionary; If the arguments are a key in this dictionary, its
        value will be returned in place of the function value
    :param dtype: The data type of the output
    :param bar: Include a progress bar?
    :return: The resulting dataframe
    """
    dct = {} if dct is None else dct
    in_ = (in_,) if isinstance(in_, str) else tuple(in_)
    dct = {((k,) if isinstance(k, str) else k): v for k, v in dct.items()}

    def row_func_(row: dict[str, object]):
        args = tuple(map(row.get, in_))
        return dct[args] if args in dct else func_(*args)

    row_iter = df.iter_rows(named=True)
    if bar:
        row_iter = tqdm(row_iter, total=df.shape[0])

    vals = list(map(row_func_, row_iter))
    if out is not None:
        col = polars.Series(name=out, values=vals)
        if dtype is not None:
            col = col.cast(dtype)

        df = df.with_columns(narwhals.from_native(col, series_only=True))

    return df


@narwhals.narwhalify
def lookup_dict(
    df: FrameT, in_: Key_ | None = None, out_: Key_ | None = None
) -> dict[Value_, Value_]:
    """Form a lookup dictionary mapping one column onto another in a dataframe.

    Allows mappings between sets of columns, using a tuple of column keys

    :param df: The dataframe
    :param in_: The input key or keys; if `None` the index is used
    :param out_: The output key or keys; if `None` the index is used
    :return: The dictionary mapping input values to output values
    """
    cols = df.columns

    def check_(key_):
        return (
            True
            if key_ is None
            else key_ in cols
            if isinstance(key_, str)
            else all(k in cols for k in key_)
        )

    def values_(key_):
        return (
            range(df.shape[0])
            if key_ is None
            else (df[key_] if isinstance(key_, str) else df[list(key_)].iter_rows())
        )

    assert check_(in_), f"{in_} not in {df}"
    assert check_(out_), f"{out_} not in {df}"

    return dict(zip(values_(in_), values_(out_), strict=True))
