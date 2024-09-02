"""Functions for writing CHEMKIN-formatted files."""

import itertools

import automol

from ..._mech import Mechanism
from ..._mech import reactions as mech_reactions
from ..._mech import species as mech_species
from ...schema import Species
from .read import KeyWord


def mechanism(mech: Mechanism, out: str | None = None) -> str:
    """Write a mechanism to CHEMKIN format.

    :param mech: A mechanism
    :param out: Optionally, write the output to this file path
    :return: The CHEMKIN mechanism as a string
    """
    reactions_block(mech)
    # blocks = [elements_block(mech), species_block(mech)]
    # mech_str = "\n\n".join(blocks)
    # print(mech_str)


def elements_block(mech: Mechanism) -> str:
    """Write the elements block to a string.

    :param mech: A mechanism
    :return: The elements block string
    """
    spc_df = mech_species(mech)
    fmls = list(map(automol.amchi.formula, spc_df[Species.amchi].to_list()))
    elems = set(itertools.chain(*(f.keys() for f in fmls)))
    elems = automol.form.sorted_symbols(elems)
    return "\n".join([KeyWord.ELEMENTS, *elems, KeyWord.END])


def species_block(mech: Mechanism) -> str:
    """Write the species block to a string.

    :param mech: A mechanism
    :return: The species block string
    """
    spc_df = mech_species(mech)
    names = spc_df[Species.name].to_list()
    return "\n".join([KeyWord.SPECIES, *names, KeyWord.END])


def reactions_block(mech: Mechanism) -> str:
    """Write the reactions block to a string.

    :param mech: A mechanism
    :return: The reactions block string
    """
    rxn_df = mech_reactions(mech)
    print(rxn_df)
