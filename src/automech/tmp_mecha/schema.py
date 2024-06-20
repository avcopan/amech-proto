from typing import Optional

import automol
import polars
from pandera import polars as pa


class Species(pa.DataFrameModel):
    name: str
    mult: int
    charge: int
    chi: str
    smi: Optional[str]
    # Original column names (before stereoexpansion)
    orig_name: Optional[str]
    orig_chi: Optional[str]
    orig_smi: Optional[str]


S_CURR_COLS = (Species.name, Species.chi, Species.smi)
S_ORIG_COLS = (Species.orig_name, Species.orig_chi, Species.orig_smi)


class Reactions(pa.DataFrameModel):
    eq: str
    rate: Optional[object]
    chi: Optional[str]
    obj: Optional[object]
    orig_eq: Optional[str]
    orig_chi: Optional[str]


R_CURR_COLS = (Reactions.eq, Reactions.chi)
R_ORIG_COLS = (Reactions.orig_eq, Reactions.orig_chi)
DUP_DIFF_COLS = (Reactions.rate, Reactions.chi, Reactions.obj)


def validate_species(df: polars.DataFrame, smi: bool = False) -> polars.DataFrame:
    """Validate a species data frame

    :param df: The dataframe
    :param smi: Add in a SMILES column?
    :return: The validated dataframe
    """
    assert (
        Species.chi in df or Species.smi in df
    ), f"Must have either 'chi' or 'smi' column: {df}"

    if Species.chi not in df:
        df[Species.chi] = df[Species.smi].map_elements(automol.smiles.chi)

    if smi and Species.smi not in df:
        df[Species.smi] = df[Species.chi].map_elements(automol.amchi.smiles)

    return validate(Species, df)


def validate_reactions(df: polars.DataFrame) -> polars.DataFrame:
    """Validate a reactions data frame

    :param df: The dataframe
    :return: The validated dataframe
    """
    return validate(Reactions, df)


def validate(model: pa.DataFrameModel, df: polars.DataFrame) -> polars.DataFrame:
    """Validate a dataframe based on a model

    :param model: The model
    :param df: The dataframe
    :return: The validated dataframe
    """
    schema = model.to_schema()
    schema.strict = False
    df = schema.validate(df)
    cols = [c for c in schema.columns.keys() if c in df]
    cols.extend(c for c in df.columns if c not in cols)
    return df[cols]
