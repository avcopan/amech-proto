"""Functions for reading RMG-formatted files."""

import os

import automol
import pandas
import pyparsing as pp
from automol.graph import RMG_ADJACENCY_LIST
from pandera.typing import DataFrame
from pyparsing import pyparsing_common as ppc
from tqdm.auto import tqdm

from ... import schema
from ...data.reac import SPECIES_NAME
from ...util import df_

MULTIPLICITY = pp.CaselessLiteral("multiplicity") + ppc.integer("mult")
SPECIES_ENTRY = (
    SPECIES_NAME("species") + pp.Opt(MULTIPLICITY) + RMG_ADJACENCY_LIST("adj_list")
)
SPECIES_DICT = pp.OneOrMore(pp.Group(SPECIES_ENTRY))("dict")


def species(inp: str, out: str | None = None) -> DataFrame[schema.Species]:
    """Extract species information as a dataframe from an RMG species dictionary.

    :param inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    inp = open(inp).read() if os.path.exists(inp) else inp

    spc_par_rets = SPECIES_DICT.parseString(inp).asDict()["dict"]

    names = []
    mults = []
    smis = []
    chis = []
    for spc_par_ret in tqdm(spc_par_rets):
        adj_par_ret = spc_par_ret["adj_list"]
        gra = automol.graph.from_parsed_rmg_adjacency_list(adj_par_ret)

        names.append(spc_par_ret["species"])
        mults.append(spc_par_ret.get("mult", 1))
        chis.append(automol.graph.inchi(gra))
        smis.append(automol.graph.smiles(gra))

    spc_df = pandas.DataFrame(
        {
            schema.Species.name: names,
            schema.Species.mult: mults,
            schema.Species.chi: chis,
            schema.Species.smi: smis,
        }
    )

    spc_df = schema.validate_species(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df
