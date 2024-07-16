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
) -> SpeciesDataFrame:
    """Validate a species data frame.

    :param df: The dataframe
    :param models: DataFrame models to validate against
    :param name_dct: If generating names, specify some names by ChI
    :param spin_dct: If generating spins, specify some spins by ChI
    :param charge_dct: If generating charges, specify some charges by ChI
    :return: The validated dataframe
    """
    dt_dct = types([Species])
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

    if Species.name not in df:
        dt = dt_dct[Species.name]
        df = df_.map_(
            df,
            Species.chi,
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
                Species.chi,
                Species.spin,
                automol.amchi.guess_spin,
                dtype=dt,
                dct=spin_dct,
            )

    if Species.charge not in df:
        dt = dt_dct[Species.charge]
        df = df_.map_(
            df, Species.chi, Species.charge, lambda _: 0, dtype=dt, dct=charge_dct
        )

    for model in models:
        df = model.validate(df)
    return df


def reaction_table(
    df: polars.DataFrame, models: Sequence[pa.DataFrameModel] = (Reaction,)
) -> ReactionDataFrame:
    """Validate a reactions data frame.

    :param df: The dataframe
    :return: The validated dataframe
    """
    df = df.rename({k: str.lower(k) for k in df.columns})

    for model in models:
        df = model.validate(df)
    return df
