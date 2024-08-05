"""DataFrame schemas."""

import itertools
from collections.abc import Sequence

import automol
import pandera.polars as pa
import polars

from automech.util import df_


# Core table schemas
class Species(pa.DataFrameModel):
    """Core species table."""

    name: str
    smiles: str
    amchi: str
    spin: int
    charge: int


class Reaction(pa.DataFrameModel):
    """Core reaction table."""

    eq: str


# Extended tables
class ReactionRate(pa.DataFrameModel):
    """Reaction table with rate."""

    rate: object


# class SpeciesExp(pa.DataFrameModel):
#     """Stereo-expanded species table."""

#     orig_name: str
#     orig_chi: str
#     orig_smi: str


def types(
    models: Sequence[pa.DataFrameModel], keys: Sequence[str] | None = None
) -> dict[str, type]:
    """Get a dictionary mapping column names to type names.

    :param *models: The models to get the schema for
    :return: The schema, as a mapping of column names onto types
    """
    keys = None if keys is None else list(keys)

    type_dct = {}
    for model in models:
        type_dct.update({k: v.dtype.type for k, v in model.to_schema().columns.items()})

    if keys is not None:
        type_dct = {k: v for k, v in type_dct.items() if k in keys}

    return type_dct


def species_table(
    df: polars.DataFrame,
    models: Sequence[pa.DataFrameModel] = (Species,),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
) -> polars.DataFrame:
    """Validate a species data frame.

    :param df: The dataframe
    :param models: DataFrame models to validate against
    :param name_dct: If generating names, specify some names by ChI
    :param spin_dct: If generating spins, specify some spins by ChI
    :param charge_dct: If generating charges, specify some charges by ChI
    :return: The validated dataframe
    """
    dt_dct = types([Species])
    df = df.rename({k: str.lower(k) for k in df.columns})
    assert (
        Species.amchi in df or Species.smiles in df
    ), f"Must have either 'chi' or 'smi' column: {df}"

    if Species.amchi not in df:
        df = df_.map_(df, Species.smiles, Species.amchi, automol.smiles.amchi)

    if Species.smiles not in df:
        df = df_.map_(df, Species.amchi, Species.smiles, automol.amchi.smiles)

    if Species.name not in df:
        dt = dt_dct[Species.name]
        df = df_.map_(
            df,
            Species.amchi,
            Species.name,
            automol.amchi.chemkin_name,
            dtype=dt,
            dct=name_dct,
        )

    if Species.spin not in df:
        dt = dt_dct[Species.spin]
        if "mult" in df:
            df = df.with_columns((df["mult"] - 1).alias(Species.spin).cast(dt))
        else:
            df = df_.map_(
                df,
                Species.amchi,
                Species.spin,
                automol.amchi.guess_spin,
                dtype=dt,
                dct=spin_dct,
            )

    if Species.charge not in df:
        dt = dt_dct[Species.charge]
        df = df_.map_(
            df, Species.amchi, Species.charge, lambda _: 0, dtype=dt, dct=charge_dct
        )

    for model in models:
        df = model.validate(df)

    cols = list(itertools.chain(*(model.to_schema().columns for model in models)))
    cols.extend(c for c in df.columns if c not in cols)
    return df.select(cols)


def reaction_table(
    df: polars.DataFrame, models: Sequence[pa.DataFrameModel] = (Reaction,)
) -> polars.DataFrame:
    """Validate a reactions data frame.

    :param df: The dataframe
    :return: The validated dataframe
    """
    df = df.rename({k: str.lower(k) for k in df.columns})

    for model in models:
        df = model.validate(df)

    cols = list(itertools.chain(*(model.to_schema().columns for model in models)))
    cols.extend(c for c in df.columns if c not in cols)
    return df.select(cols)
