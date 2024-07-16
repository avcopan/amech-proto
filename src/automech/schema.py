"""DataFrame schemas."""

from collections.abc import Sequence

import automol
import pandera.polars as pa
import polars
from pandera.typing.polars import DataFrame

from automech.util import df_


# Core table schemas
class Species(pa.DataFrameModel):
    """Core species table."""

    name: str
    spin: int
    charge: int
    chi: str
    smi: str


SpeciesDataFrame = DataFrame[Species]


class Reaction(pa.DataFrameModel):
    """Core reaction table."""

    eq: str


ReactionDataFrame = DataFrame[Species]


# Extended tables
class ReactionRate(Reaction):
    """Reaction table with rate."""

    rate: object


# class SpeciesExp(Species):
#     """Stereo-expanded species table."""

#     orig_name: str
#     orig_chi: str
#     orig_smi: str


def types(*models: pa.DataFrameModel) -> dict[str, type]:
    """Get a dictionary mapping column names to type names.

    :param *models: The models to get the schema for
    :return: The schema, as a mapping of column names onto types
    """
    type_dct = {}
    for model in models:
        type_dct.update({k: v.dtype.type for k, v in model.to_schema().columns.items()})
    return type_dct


def species_table(
    df: polars.DataFrame, models: Sequence[pa.DataFrameModel] = (Species,)
) -> SpeciesDataFrame:
    """Validate a species data frame.

    :param df: The dataframe
    :param smi: Add in a SMILES column?
    :param models: DataFrame models to validate against
    :return: The validated dataframe
    """
    dt_dct = types(Species)
    rename_dct = {"smiles": Species.smi, "inchi": Species.chi}
    df = df.rename({k: str.lower(k) for k in df.columns})
    df = df.rename({k: v for k, v in rename_dct.items() if k in df})

    assert (
        Species.chi in df or Species.smi in df
    ), f"Must have either 'chi' or 'smi' column: {df}"

    if Species.chi not in df:
        df = df_.map_(df, Species.smi, Species.chi, automol.smiles.amchi)

    if Species.smi not in df:
        df = df_.map_(df, Species.chi, Species.smi, automol.amchi.smiles)

    if Species.spin not in df:
        dt = dt_dct[Species.spin]
        if "mult" in df:
            df = df.with_columns((df["mult"] - 1).alias(Species.spin).cast(dt))
        else:
            df = df_.map_(
                df, Species.chi, Species.spin, automol.amchi.guess_spin, dtype=dt
            )

    if Species.charge not in df:
        dt = dt_dct[Species.charge]
        df = df.with_columns(polars.lit(0).alias(Species.charge).cast(dt))

    print(df)
    for model in models:
        df = df.pipe(model.validate)
    return df


def reaction_table(
    df: polars.DataFrame, models: Sequence[pa.DataFrameModel] = (Reaction,)
) -> ReactionDataFrame:
    """Validate a reactions data frame.

    :param df: The dataframe
    :return: The validated dataframe
    """
    df = df.rename({k: str.lower(k) for k in df.columns})
    print("validating...")
    for model in models:
        print(type(df))
        df = model.validate(df)
    return df
