"""Functions for reading Mechanalyzer-formatted files."""

import io
import os

import polars

from ..._mech import Mechanism
from ..._mech import from_data as mechanism_from_data
from ...schema import species_table
from ...util import df_
from ..chemkin import read as chemkin_read


def mechanism(
    inp: str, spc_inp: str, out: str | None = None, spc_out: str | None = None
) -> Mechanism:
    """Extract the mechanism from RMG files.

    :param inp: An RMG mechanism (CHEMKIN format), as a file path or string
    :param spc_inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    rxn_df = chemkin_read.reactions(inp, out=out)
    spc_df = species(spc_inp, out=spc_out)
    return mechanism_from_data(inp=rxn_df, spc_inp=spc_df)


def species(inp: str, out: str | None = None) -> polars.DataFrame:
    """Extract species information as a dataframe from a Mechanalyzer species CSV.

    :param inp: A Mechanalyzer species CSV, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    inp = open(inp).read() if os.path.exists(inp) else inp

    spc_df = polars.read_csv(io.StringIO(inp), quote_char="'")
    spc_df = species_table(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df
