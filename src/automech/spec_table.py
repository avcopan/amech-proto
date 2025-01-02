"""Functions acting on species DataFrames."""

from collections.abc import Sequence

import automol
import polars

from .schema import Species
from .util import df_


# selections
def by_id(
    spc_df: polars.DataFrame,
    species: Sequence[str] | None = None,
    species_id: str | Sequence[str] = Species.name,
) -> polars.DataFrame:
    """Add a column indicating a match against one or more species.

    :param spc_df: A species DataFrame
    :param col_name: The column name
    :param species: Species identifiers
    :param species_id: One or more columns for identifying species
    :return: The modified species DataFrame
    """
    col_match = df_.temp_column()
    spc_df = with_species_match_column(
        spc_df, col_match, species=species, species_id=species_id
    )
    spc_df = spc_df.filter(col_match)
    return spc_df.drop(col_match)


# transformations
def with_species_match_column(
    spc_df: polars.DataFrame,
    col_name: str,
    species: Sequence[str] | None = None,
    species_id: str | Sequence[str] = Species.name,
) -> polars.DataFrame:
    """Add a column indicating a match against one or more species.

    :param spc_df: A species DataFrame
    :param col_name: The column name
    :param species: Species identifiers
    :param species_id: One or more columns for identifying species
    :return: The modified species DataFrame
    """
    if species is None:
        return spc_df.with_columns(polars.lit(True).alias(col_name))

    if isinstance(species_id, str):
        species_id = [species_id]
        species = [[s] for s in species]

    match_data = dict(zip(species_id, zip(*species, strict=True), strict=True))

    # Replace SMILES with AMChI if given
    if Species.smiles in match_data:
        match_data[Species.amchi] = list(
            map(automol.smiles.amchi, match_data.pop(Species.smiles))
        )

    match_df = polars.DataFrame({col_name: True, **match_data})
    return spc_df.join(match_df, on=match_data.keys(), how="left").with_columns(
        polars.col(col_name).fill_null(False)
    )
