"""DataFrame utilities."""

import random
import string
from collections.abc import Callable, Sequence
from pathlib import Path

import polars
from tqdm.auto import tqdm

Key = str
Keys = Sequence[str]
Key_ = Key | Keys
Value = object
Values = Sequence[object]
Value_ = Value | Values


def with_index(df: polars.DataFrame, name: str = "index") -> polars.DataFrame:
    return df.with_row_index(name=name)


def temp_column(length: int = 24) -> str:
    """Generate a unique temporary column name for a dataframe.

    :return: The column name
    """
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


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
    kwargs = (
        {}
        if quote_char is None
        else {"quote_char": quote_char, "quote_style": "non_numeric"}
    )
    if path is not None:
        path: Path = Path(path)
        df.write_csv(path, **kwargs)


def map_(
    df: polars.DataFrame,
    in_: Key_,
    out_: Key_ | None,
    func_: Callable,
    dct: dict[object, object] | None = None,
    dtype_: polars.DataType | Sequence[polars.DataType] | None = None,
    bar: bool = True,
) -> polars.DataFrame:
    """Map columns from a dataframe onto a new column.

    :param df: The dataframe
    :param in_: The input key or keys
    :param out_: The output key or keys; if `None`, the function output will be ignored
    :param func_: The mapping function
    :param dct: A lookup dictionary; If the arguments are a key in this dictionary, its
        value will be returned in place of the function value
    :param dtype_: The data type of the output
    :param bar: Include a progress bar?
    :return: The resulting dataframe
    """
    dct = {} if dct is None else dct
    in_ = (in_,) if isinstance(in_, str) else tuple(in_)
    dct = {((k,) if isinstance(k, str) else k): v for k, v in dct.items()}

    def row_func_(row: dict[str, object]):
        args = tuple(map(row.get, in_))
        return dct[args] if dct and args in dct else func_(*args)

    row_iter = df.iter_rows(named=True)
    if bar:
        row_iter = tqdm(row_iter, total=df.shape[0])

    vals = list(map(row_func_, row_iter))
    if out_ is None:
        return

    # Post-process the output
    if isinstance(out_, str):
        out_ = (out_,)
        dtype_ = (dtype_,)
        vals_lst = (vals,)
    else:
        vals_lst = list(zip(*vals, strict=True))
        nvals = len(vals_lst)
        assert len(out_) == nvals, f"Cannot match {nvals} output values to keys {out_}"
        dtype_ = (None,) * nvals if dtype_ is None else dtype_

    assert len(out_) == len(dtype_), f"Cannot match {dtype_} dtypes to keys {out_}"

    for out, dtype, vals in zip(out_, dtype_, vals_lst, strict=True):
        col = polars.Series(name=out, values=vals, dtype=dtype, strict=False)
        df = df.with_columns(col)

    return df


def lookup_dict(
    df: polars.DataFrame, in_: Key_ | None = None, out_: Key_ | None = None
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
            else (
                df[key_].to_list()
                if isinstance(key_, str)
                else df[list(key_)].iter_rows()
            )
        )

    assert check_(in_), f"{in_} not in {df}"
    assert check_(out_), f"{out_} not in {df}"

    return dict(zip(values_(in_), values_(out_), strict=True))
