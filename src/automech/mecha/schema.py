from typing import Optional

import automol
import pandas
import pandera as pa
from pandera.typing import Series
from tqdm.auto import tqdm

tqdm.pandas()


class Species(pa.DataFrameModel):
    name: Series[str] = pa.Field(coerce=True)
    mult: Series[int] = pa.Field(coerce=True)
    charge: Series[int] = pa.Field(coerce=True, default=0)
    chi: Series[str]
    smi: Optional[Series[str]]
    # Original column names (before stereoexpansion)
    orig_name: Optional[Series[str]]
    orig_chi: Optional[Series[str]]
    orig_smi: Optional[Series[str]]


S_CURR_COLS = (Species.name, Species.chi, Species.smi)
S_ORIG_COLS = (Species.orig_name, Species.orig_chi, Species.orig_smi)


class Reactions(pa.DataFrameModel):
    eq: Series[str] = pa.Field(coerce=True)
    rate: Optional[Series[object]]
    chi: Optional[Series[str]]
    obj: Optional[Series[object]]
    orig_eq: Optional[Series[str]]
    orig_chi: Optional[Series[str]]


R_CURR_COLS = (Reactions.eq, Reactions.chi)
R_ORIG_COLS = (Reactions.orig_eq, Reactions.orig_chi)
DUP_DIFF_COLS = (Reactions.rate, Reactions.chi, Reactions.obj)


def validate_species(df: pandas.DataFrame, smi: bool = False) -> pandas.DataFrame:
    """Validate a species data frame

    :param df: The dataframe
    :param smi: Add in a SMILES column?
    :return: The validated dataframe
    """
    assert (
        Species.chi in df or Species.smi in df
    ), f"Must have either 'chi' or 'smi' column: {df}"

    if Species.chi not in df:
        df[Species.chi] = df[Species.smi].progress_apply(automol.smiles.chi)

    if smi and Species.smi not in df:
        df[Species.smi] = df[Species.chi].progress_apply(automol.amchi.smiles)

    return validate(Species, df)


def validate_reactions(df: pandas.DataFrame) -> pandas.DataFrame:
    """Validate a reactions data frame

    :param df: The dataframe
    :return: The validated dataframe
    """
    return validate(Reactions, df)


def validate(model: pa.DataFrameModel, df: pandas.DataFrame) -> pandas.DataFrame:
    """Validate a pandas dataframe based on a model

    :param model: The model
    :param df: The dataframe
    :return: The validated dataframe
    """
    schema = model.to_schema()
    schema.add_missing_columns = True
    schema.strict = False
    df = schema.validate(df)
    cols = [c for c in schema.columns.keys() if c in df]
    cols.extend(df.columns.difference(cols))
    return df[cols]
