"""DataFrame schemas."""

import automol
import pandas
import pandera as pa
from pandera.typing import DataFrame, Series
from tqdm.auto import tqdm

tqdm.pandas()


class Species(pa.DataFrameModel):
    name: Series[str] = pa.Field(coerce=True)
    spin: Series[int] = pa.Field(coerce=True)
    charge: Series[int] = pa.Field(coerce=True, default=0)
    chi: Series[str]
    smi: Series[str] | None
    # Original column names (before stereoexpansion)
    orig_name: Series[str] | None
    orig_chi: Series[str] | None
    orig_smi: Series[str] | None


S_CURR_COLS = (Species.name, Species.chi, Species.smi)
S_ORIG_COLS = (Species.orig_name, Species.orig_chi, Species.orig_smi)


class Reactions(pa.DataFrameModel):
    eq: Series[str] = pa.Field(coerce=True)
    rate: Series[object] | None
    chi: Series[str] | None
    obj: Series[object] | None
    orig_eq: Series[str] | None
    orig_chi: Series[str] | None


R_CURR_COLS = (Reactions.eq, Reactions.chi)
R_ORIG_COLS = (Reactions.orig_eq, Reactions.orig_chi)
DUP_DIFF_COLS = (Reactions.rate, Reactions.chi, Reactions.obj)


def validate_species(df: DataFrame, smi: bool = False) -> DataFrame[Species]:
    """Validate a species data frame.

    :param df: The dataframe
    :param smi: Add in a SMILES column?
    :return: The validated dataframe
    """
    rename_dct = {"smiles": Species.smi, "inchi": Species.chi}
    df = df.rename(str.lower, axis="columns")
    df = df.rename(rename_dct, axis="columns")

    assert (
        Species.chi in df or Species.smi in df
    ), f"Must have either 'chi' or 'smi' column: {df}"

    if Species.chi not in df:
        df[Species.chi] = df[Species.smi].progress_apply(automol.smiles.chi)

    if smi and Species.smi not in df:
        df[Species.smi] = df[Species.chi].progress_apply(automol.amchi.smiles)

    if Species.spin not in df:
        df[Species.spin] = (
            df["mult"] - 1
            if "mult" in df
            else df[Species.chi].apply(automol.amchi.guess_spin)
        )

    return validate(Species, df)


def validate_reactions(df: DataFrame) -> DataFrame[Species]:
    """Validate a reactions data frame.

    :param df: The dataframe
    :return: The validated dataframe
    """
    return validate(Reactions, df)


def validate(model: pa.DataFrameModel, df: pandas.DataFrame) -> pandas.DataFrame:
    """Validate a pandas dataframe based on a model.

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
