"""Functions for reading RMG-formatted files."""

import os
from pathlib import Path

import automol
import polars
import pyparsing as pp
from automol.graph import RMG_ADJACENCY_LIST
from pyparsing import pyparsing_common as ppc
from tqdm.auto import tqdm

from ... import schema
from ..._mech import Mechanism
from ..._mech import from_data as mechanism_from_data
from ...schema import Errors, Species
from ...util import df_
from ..chemkin import read as chemkin_read

MULTIPLICITY = pp.CaselessLiteral("multiplicity") + ppc.integer("mult")
SPECIES_NAME = pp.Word(pp.printables)
SPECIES_ENTRY = (
    SPECIES_NAME("species") + pp.Opt(MULTIPLICITY) + RMG_ADJACENCY_LIST("adj_list")
)
SPECIES_DICT = pp.OneOrMore(pp.Group(SPECIES_ENTRY))("dict")


def mechanism(
    rxn_inp: str, spc_inp: str, out: str | None = None, spc_out: str | None = None
) -> tuple[Mechanism, Errors]:
    """Extract the mechanism from RMG files.

    :param inp: An RMG mechanism (CHEMKIN format), as a file path or string
    :param spc_inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    spc_df = species(spc_inp)
    spc_df = chemkin_read.thermo(rxn_inp, spc_df=spc_df, out=spc_out)
    rxn_df, err = chemkin_read.reactions(rxn_inp, out=out, spc_df=spc_df)
    rate_units = chemkin_read.reactions_units(rxn_inp)
    thermo_temps = chemkin_read.thermo_temperatures(rxn_inp)
    mech = mechanism_from_data(
        rxn_inp=rxn_df, spc_inp=spc_df, rate_units=rate_units, thermo_temps=thermo_temps
    )
    return mech, err


def species(inp: str, out: str | None = None) -> polars.DataFrame:
    """Extract species information as a dataframe from an RMG species dictionary.

    :param inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)

    spc_par_rets = SPECIES_DICT.parse_string(inp, parse_all=True).asDict()["dict"]

    names = []
    mults = []
    smis = []
    chis = []
    for spc_par_ret in tqdm(spc_par_rets):
        adj_par_ret = spc_par_ret["adj_list"]
        gra = automol.graph.from_parsed_rmg_adjacency_list(adj_par_ret)

        names.append(spc_par_ret["species"])
        mults.append(spc_par_ret.get("mult", 1) - 1)
        chis.append(automol.graph.amchi(gra))
        smis.append(automol.graph.smiles(gra))

    data_dct = {
        Species.name: names,
        Species.spin: mults,
        Species.amchi: chis,
        Species.smiles: smis,
    }
    spc_df = polars.DataFrame(
        data=data_dct, schema=schema.species_types(data_dct.keys())
    )

    spc_df = schema.species_table(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df
