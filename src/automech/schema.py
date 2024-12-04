"""DataFrame schemas."""

import itertools
from collections.abc import Sequence

import automol
import pandera.polars as pa
import polars
from polars.datatypes import Struct

from automech.util import df_

Model = pa.DataFrameModel


# Core table schemas
class Species(Model):
    """Core species table."""

    name: str
    smiles: str
    amchi: str
    spin: int
    charge: int
    formula: Struct


class Reaction(Model):
    """Core reaction table."""

    reactants: list[str]
    products: list[str]
    # formula: Struct


# Extended tables
class ReactionRate(Model):
    """Reaction table with rate."""

    rate: Struct
    colliders: Struct = pa.Field(nullable=True)


class SpeciesThermo(Model):
    """Species table with thermo."""

    thermo_string: str


class SpeciesRenamed(Model):
    """Renamed species table."""

    orig_name: str


class ReactionRenamed(Model):
    """Renamed reaction table."""

    orig_reactants: str
    orig_products: str


class SpeciesStereo(Model):
    """Stereo-expanded species table."""

    orig_name: str
    orig_smiles: str
    orig_amchi: str


class ReactionStereo(Model):
    """Stereo-expanded reaction table."""

    amchi: str
    orig_reactants: str
    orig_products: str


class ReactionMisc(Model):
    """Miscellaneous reaction columns (not for validation)."""

    orig_rate: Struct  # Add this to `ReactionStereo` instead?


def types(
    models: Sequence[Model], keys: Sequence[str] | None = None
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
    models: Sequence[Model] = (),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
    keep_extra: bool = True,
) -> polars.DataFrame:
    """Validate a species data frame.

    :param df: The dataframe
    :param models: Extra species models to validate against
    :param name_dct: If generating names, specify some names by ChI
    :param spin_dct: If generating spins, specify some spins by ChI
    :param charge_dct: If generating charges, specify some charges by ChI
    :return: The validated dataframe
    """
    if Species not in models:
        models = (Species, *models)

    dt_dct = types([Species])
    df = df.rename({k: str.lower(k) for k in df.columns})
    assert (
        Species.amchi in df or Species.smiles in df
    ), f"Must have either 'amchi' or 'smiles' column: {df}"

    # Sanitize SMILES, which may have things like 'singlet[CH2]'
    spin_smi_dct = {"singlet": 0, "triplet": 2}
    if Species.smiles in df:
        for spin_type in spin_smi_dct:
            df = df.with_columns(
                polars.col(Species.smiles).str.contains(spin_type).alias(spin_type)
            )
            df = df.with_columns(polars.col(Species.smiles).str.replace(spin_type, ""))

    if Species.amchi not in df:
        df = df_.map_(df, Species.smiles, Species.amchi, automol.smiles.amchi)

    if Species.smiles not in df:
        df = df_.map_(df, Species.amchi, Species.smiles, automol.amchi.smiles)

    if Species.formula not in df:
        dt = dt_dct[Species.formula]
        df = df_.map_(
            df, Species.amchi, Species.formula, automol.amchi.formula, dtype_=dt
        )

    if Species.name not in df:
        dt = dt_dct[Species.name]
        df = df_.map_(
            df,
            Species.amchi,
            Species.name,
            automol.amchi.chemkin_name,
            dtype_=dt,
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
                dtype_=dt,
                dct=spin_dct,
            )

    for spin_type, spin_val in spin_smi_dct.items():
        if spin_type in df:
            df = df.with_columns(
                spin=polars.when(polars.col(spin_type))
                .then(spin_val)
                .otherwise(polars.col(Species.spin))
            )
            df = df.drop(spin_type)

    if Species.charge not in df:
        dt = dt_dct[Species.charge]
        df = df_.map_(
            df, Species.amchi, Species.charge, lambda _: 0, dtype_=dt, dct=charge_dct
        )

    for model in models:
        df = model.validate(df)

    return table_with_columns_from_models(df, models=models, keep_extra=keep_extra)


def reaction_table(
    df: polars.DataFrame, models: Sequence[Model] = (), keep_extra: bool = True
) -> polars.DataFrame:
    """Validate a reactions data frame.

    :param df: The dataframe
    :param models: Extra reaction models to validate against
    :return: The validated dataframe
    """
    if Reaction not in models:
        models = (Reaction, *models)

    df = df.rename({k: str.lower(k) for k in df.columns})

    for model in models:
        df = model.validate(df)

    return table_with_columns_from_models(df, models=models, keep_extra=keep_extra)


def table_with_columns_from_models(
    df: polars.DataFrame, models: Sequence[Model] = (), keep_extra: bool = True
) -> polars.DataFrame:
    """Return a table with columns selected from models.

    :param df: The dataframe
    :param models: The models, defaults to ()
    :param drop_extra: Keep extra columns that aren't in the models?
    :return: The dataframe selection
    """
    cols = list(itertools.chain(*(model.to_schema().columns for model in models)))
    if keep_extra:
        cols.extend(c for c in df.columns if c not in cols)
    return df.select(cols)
