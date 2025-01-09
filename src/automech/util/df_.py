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


def count(df: polars.DataFrame) -> int:
    """Count the number of rows in a DataFrame.

    :param df: The DataFrame
    :return: The number of rows
    """
    return df.select(polars.len()).item()


def with_index(df: polars.DataFrame, col: str = "index") -> polars.DataFrame:
    """Add index column to DataFrame.

    :param df: DataFrame
    :param col: _description_, defaults to "index"
    :return: _description_
    """
    return df.with_row_index(name=col)


def with_intersection_columns(
    df1: polars.DataFrame,
    df2: polars.DataFrame,
    comp_col_: str | list[str],
    comp_col2_: str | list[str] | None = None,
    col: str = "intersection",
) -> tuple[polars.DataFrame, polars.DataFrame]:
    """Add columns to DataFrame pair indicating their intersection.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param comp_col: Column to compare
    :param comp_col2: Column from second DataFrame to compare, if different
    :param col: Name of intersection column
    :return: First and second DataFrames with intersection columns
    """
    comp_col2_ = comp_col_ if comp_col2_ is None else comp_col2_
    df1 = with_intersection_column(
        df1, df2, comp_col_=comp_col_, comp_col2_=comp_col2_, col=col
    )
    df2 = with_intersection_column(
        df2, df1, comp_col_=comp_col2_, comp_col2_=comp_col_, col=col
    )
    return df1, df2


def with_intersection_column(
    df1: polars.DataFrame,
    df2: polars.DataFrame,
    comp_col_: str | list[str],
    comp_col2_: str | list[str] | None = None,
    col: str = "intersection",
) -> polars.DataFrame:
    """Add a column indicating intersection with another DataFrame.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param comp_col: Column to compare
    :param comp_col2: Column from second DataFrame to compare, if different
    :param col: Name of intersection column
    :return: First DataFrame with intersection column
    """
    comp_col2_ = comp_col_ if comp_col2_ is None else comp_col2_
    comp_col_ = [comp_col_] if isinstance(comp_col_, str) else comp_col_
    comp_col2_ = [comp_col2_] if isinstance(comp_col2_, str) else comp_col2_
    col_dct = dict(zip(comp_col_, comp_col2_, strict=True))

    df1 = df1.with_columns(
        polars.all_horizontal(
            *(polars.col(c1).is_in(df2.get_column(c2)) for c1, c2 in col_dct.items())
        ).alias(col)
    )
    return df1


def has_values(df: polars.DataFrame) -> bool:
    """Determine if DataFrame has non-null values.

    :param df: DataFrame
    :return: `True` if it does, `False` if it doesn't
    """
    return df.select(polars.any_horizontal(polars.col("*").is_not_null().any())).item()


def temp_column(length: int = 24) -> str:
    """Generate a unique temporary column name for a DataFrame.

    :param length: The length of the temporary column name, defaults to 24
    :return: The column name
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def from_csv(path: str) -> polars.DataFrame:
    """Read a DataFrame from a CSV file.

    :param path: The path to the CSV file
    :return: The DataFrame
    """
    try:
        df = polars.read_csv(path)
    except polars.exceptions.ComputeError:
        df = polars.read_csv(path, quote_char="'")
    return df


def to_csv(
    df: polars.DataFrame, path: str | None, quote_char: str | None = None
) -> None:
    """Write a DataFrame to a CSV file.

    If `path` is `None`, this function does nothing.

    :param df: The DataFrame
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
    bar: bool = False,
) -> polars.DataFrame:
    """Map columns from a DataFrame onto a new column.

    :param df: The DataFrame
    :param in_: The input key or keys
    :param out_: The output key or keys; if `None`, the function output will be ignored
    :param func_: The mapping function
    :param dct: A lookup dictionary; if the arguments are a key in this dictionary, its
        value will be returned in place of the function value
    :param dtype_: The data type of the output
    :param bar: Include a progress bar?
    :return: The resulting DataFrame
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
        df = df.with_columns(
            polars.Series(name=out, values=vals, dtype=dtype, strict=False)
        )

    return df


def lookup_dict(
    df: polars.DataFrame, in_: Key_ | None = None, out_: Key_ | None = None
) -> dict[Value_, Value_]:
    """Form a lookup dictionary mapping one column onto another in a DataFrame.

    Allows mappings between sets of columns, using a tuple of column keys.

    :param df: The DataFrame
    :param in_: The input key or keys; if `None`, the index is used
    :param out_: The output key or keys; if `None`, the index is used
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
