"""Functions acting on species DataFrames."""

from collections.abc import Mapping, Sequence

import automol
import polars

from . import schema
from .schema import Species, SpeciesMisc, SpeciesThermo
from .util import col_, df_

m_col_ = col_

SPECIES_KEY_COLS = (Species.amchi, Species.spin, Species.charge)


# update
def left_update(
    spc_df: polars.DataFrame,
    src_spc_df: polars.DataFrame,
    key_col_: str | Sequence[str] = SPECIES_KEY_COLS,
    drop_orig: bool = False,
) -> polars.DataFrame:
    """Left-update species data by species key.

    :param spc_df: Species DataFrame
    :param src_spc_df: Source species DataFrame
    :param key_col_: Species key column(s)
    :param drop_orig: Whether to drop the original column values
    :return: Species DataFrame
    """
    # Update
    spc_df = df_.left_update(spc_df, src_spc_df, col_=key_col_, drop_orig=drop_orig)

    # Drop unnecessary columns
    drop_cols = [m_col_.orig(c) for c in schema.columns(Species) if c != Species.name]
    spc_df = spc_df.drop(drop_cols, strict=False)
    return spc_df


def left_update_thermo(
    spc_df: polars.DataFrame, src_spc_df: polars.DataFrame
) -> polars.DataFrame:
    """Read thermochemical data from one dataframe into another.

    (AVC note: I think this can be deprecated...)

    :param spc_df: Species DataFrame
    :param src_spc_df: Species DataFrame with thermochemical data
    :return: Species DataFrame
    """
    spc_df = spc_df.rename(
        {SpeciesThermo.thermo_string: SpeciesMisc.orig_thermo_string}, strict=False
    )
    spc_df = spc_df.join(src_spc_df, how="left", on=Species.name)
    return schema.species_table(spc_df, model_=SpeciesThermo)


# add columns
def with_species_key(
    spc_df: polars.DataFrame,
    col: str = "key",
    key_col_: str | Sequence[str] = SPECIES_KEY_COLS,
) -> polars.DataFrame:
    """Add a key for identifying unique species.

    The key is "{AMChI}_{spin}_{charge}"

    :param spc_df: Species DataFrame
    :param col: Column name, defaults to "key"
    :return: Species DataFrame
    """
    return df_.with_concat_string_column(spc_df, col=col, src_col_=key_col_)


# tranform
def rename(
    spc_df: polars.DataFrame,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    drop_orig: bool = False,
) -> polars.DataFrame:
    """Rename species in a species DataFrame.

    :param rxn_df: Species DataFrame
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :return: Species DataFrame
    """
    col_dct = col_.to_orig(Species.name)
    spc_df = spc_df.with_columns(polars.col(c0).alias(c) for c0, c in col_dct.items())
    expr = polars.col(Species.name)
    expr = expr.replace(names) if new_names is None else expr.replace(names, new_names)
    spc_df = spc_df.with_columns(expr)
    if drop_orig:
        spc_df = spc_df.drop(col_dct.values())
    return spc_df


# sort
def sort_by_formula(
    spc_df: polars.DataFrame, key: str = Species.formula
) -> polars.DataFrame:
    """Sort species by formula.

    :param spc_df: Species DataFrame
    :param key: Formula column key
    :return: Species DataFrame, sorted by formula
    """
    all_atoms = [s for s, *_ in spc_df.schema[key]]
    heavy_atoms = [s for s in all_atoms if s != "H"]
    return spc_df.sort(
        polars.sum_horizontal(
            polars.col(Species.formula).struct.field(*heavy_atoms)
        ),  # heavy atoms
        polars.sum_horizontal(
            polars.col(Species.formula).struct.field(*all_atoms)
        ),  # all atoms
        polars.col(Species.formula),
        nulls_last=True,
    )


# select
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
