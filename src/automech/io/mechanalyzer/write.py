"""Functions for writing Mechanalyzer-formatted files."""

import automol
import pandas
import polars

from ..._mech import Mechanism
from ..._mech import species as mech_species
from ...schema import Model, Species, table_with_columns_from_models
from ...util import df_
from ..chemkin import write as chemkin_write


class MASpecies(Model):
    """Mechanalyzer species table."""

    name: str
    smiles: str
    inchi: str
    inchikey: str
    mult: str
    charge: str
    canon_enant_ich: str


def mechanism(
    mech: Mechanism, rxn_out: str | None = None, spc_out: str | None = None
) -> tuple[dict, pandas.DataFrame]:
    """Write mechanism to MechAnalyzer format.

    :param mech: A mechanism
    :param rxn_out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: MechaAnalyzer reaction dictionary and species dataframes
    """
    chemkin_write.mechanism(mech, out=rxn_out)
    # spc_df = species(mech, out=spc_out)
    # print(spc_df)


def species(mech: Mechanism, out: str | None = None) -> pandas.DataFrame:
    """Write the species in a mechanism to a MechAnalyzer species table.

    :param mech: A mechanism
    :param out: Optionally, write the output to this file path (species)
    :return: A MechAnalyzer species dataframe
    """
    # Write species
    spc_df = mech_species(mech)
    spc_df = df_.map_(spc_df, Species.amchi, MASpecies.inchi, automol.amchi.chi_)
    spc_df = df_.map_(
        spc_df,
        MASpecies.inchi,
        MASpecies.canon_enant_ich,
        automol.amchi.canonical_enantiomer,
    )
    spc_df = spc_df.with_columns((polars.col(Species.spin) + 1).alias(MASpecies.mult))
    spc_df = table_with_columns_from_models(
        spc_df, models=[MASpecies], keep_extra=False
    )
    if out is not None:
        df_.to_csv(spc_df, out, quote_char="'")

    return spc_df.to_pandas()