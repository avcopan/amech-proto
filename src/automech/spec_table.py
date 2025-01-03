"""Functions acting on species DataFrames."""

from collections.abc import Sequence

import automol
import polars

from . import schema
from .schema import Species
from .util import df_


# selections
def rows_dict(
    spc_df: polars.DataFrame,
    vals_: object | Sequence[object] | None = None,
    key_: str | Sequence[str] = Species.name,
    try_fill: bool = False,
    fail_if_multiple: bool = True,
) -> dict[str, dict[str, object]]:
    """Select a row that matches a species.

    :param spc_df: A species DataFrame
    :param vals_: Column value(s) list to select
    :param key_: Column key(s) to select by
    :param try_fill: Whether attempt to fill missing values
    :param fail_if_multiple: Whether to fail if multiple matches are found
    :return: The modified species DataFrame
    """
    return {
        r.get(Species.name): r
        for r in rows(
            spc_df, vals_, key_, try_fill=try_fill, fail_if_multiple=fail_if_multiple
        )
    }


def rows(
    spc_df: polars.DataFrame,
    vals_: object | Sequence[object] | None = None,
    key_: str | Sequence[str] = Species.name,
    try_fill: bool = False,
    fail_if_multiple: bool = True,
) -> list[dict[str, object]]:
    """Select a row that matches a species.

    :param spc_df: A species DataFrame
    :param vals_: Column value(s) list to select
    :param key_: Column key(s) to select by
    :param try_fill: Whether attempt to fill missing values
    :param fail_if_multiple: Whether to fail if multiple matches are found
    :return: The modified species DataFrame
    """
    if vals_ is None:
        return spc_df.rows(named=True)

    return [
        row(spc_df, v, key_, try_fill=try_fill, fail_if_multiple=fail_if_multiple)
        for v in vals_
    ]


def row(
    spc_df: polars.DataFrame,
    val_: object | Sequence[object],
    key_: str | Sequence[str] = Species.name,
    try_fill: bool = False,
    fail_if_multiple: bool = True,
) -> dict[str, object]:
    """Select a row that matches a species.

    :param spc_df: A species DataFrame
    :param val_: Column value(s)
    :param key_: Column key(s)
    :param try_fill: Whether attempt to fill missing values
    :param fail_if_multiple: Whether to fail if multiple matches are found
    :return: The modified species DataFrame
    """
    if isinstance(key_, str):
        key_ = [key_]
        val_ = [val_]

    match_df = filter(spc_df, [val_], key_)
    count = df_.count(match_df)

    if fail_if_multiple and count > 1:
        raise ValueError(f"Multiple species match the criteria: {val_}")

    if try_fill and not count:
        data = {k: [v] for k, v in zip(key_, val_, strict=True)}
        match_df = polars.DataFrame(data)
        match_df = schema.species_table(match_df)
        match_df = polars.DataFrame(
            match_df, schema={k: spc_df.schema[k] for k in match_df.columns}
        )

    return None if match_df.is_empty() else match_df.row(0, named=True)


def filter(  # noqa: A001
    spc_df: polars.DataFrame,
    vals_: Sequence[object | Sequence[object]] | None = None,
    key_: str | Sequence[str] = Species.name,
) -> polars.DataFrame:
    """Filter to include only rows that match one or more species.

    :param spc_df: A species DataFrame
    :param col_name: The column name
    :param vals_lst: Column values list
    :param keys: Column keys
    :return: The modified species DataFrame
    """
    match_exprs = [species_match_expression(val_, key_) for val_ in vals_]
    return spc_df.filter(polars.any_horizontal(*match_exprs))


# helpers
def species_match_expression(
    val_: object | Sequence[object],
    key_: str | Sequence[str] = Species.name,
) -> polars.Expr:
    """Prepare a dictionary of species match data.

    :param val_: Column values
    :param key_: Column keys
    """
    if isinstance(key_, str):
        key_ = [key_]
        val_ = [val_]

    match_data = dict(zip(key_, val_, strict=True))
    if Species.smiles in match_data:
        match_data[Species.amchi] = automol.smiles.amchi(match_data.pop(Species.smiles))

    return polars.all_horizontal(*(polars.col(k) == v for k, v in match_data.items()))
