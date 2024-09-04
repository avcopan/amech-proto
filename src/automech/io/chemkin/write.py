"""Functions for writing CHEMKIN-formatted files."""

import itertools
from pathlib import Path

import automol

from ..._mech import Mechanism
from ..._mech import reactions as mech_reactions
from ..._mech import species as mech_species
from ...data import reac
from ...schema import Reaction, ReactionRate, Species
from .read import KeyWord


def mechanism(mech: Mechanism, out: str | None = None) -> str:
    """Write a mechanism to CHEMKIN format.

    :param mech: A mechanism
    :param out: Optionally, write the output to this file path
    :return: The CHEMKIN mechanism as a string
    """
    blocks = [elements_block(mech), species_block(mech), reactions_block(mech)]
    mech_str = "\n\n\n".join(blocks)
    if out is not None:
        out: Path = Path(out)
        out.write_text(mech_str)

    return mech_str


def elements_block(mech: Mechanism) -> str:
    """Write the elements block to a string.

    :param mech: A mechanism
    :return: The elements block string
    """
    spc_df = mech_species(mech)
    fmls = list(map(automol.amchi.formula, spc_df[Species.amchi].to_list()))
    elem_strs = set(itertools.chain(*(f.keys() for f in fmls)))
    elem_strs = automol.form.sorted_symbols(elem_strs)
    return block(KeyWord.ELEMENTS, elem_strs)


def species_block(mech: Mechanism) -> str:
    """Write the species block to a string.

    :param mech: A mechanism
    :return: The species block string
    """
    spc_df = mech_species(mech)
    name_width = 1 + spc_df[Species.name].str.len_chars().max()
    smi_width = 1 + spc_df[Species.smiles].str.len_chars().max()
    spc_strs = [
        f"{n:<{name_width}} ! SMILES: {s:<{smi_width}} AMChI: {c}"
        for n, s, c in spc_df.select(Species.name, Species.smiles, Species.amchi).rows()
    ]
    return block(KeyWord.SPECIES, spc_strs)


def reactions_block(mech: Mechanism) -> str:
    """Write the reactions block to a string.

    :param mech: A mechanism
    :return: The reactions block string
    """
    rxn_df = mech_reactions(mech)
    eq_width = 10 + rxn_df[Reaction.eq].str.len_chars().max()
    rxn_strs = [
        reac.chemkin_string(reac.from_equation(eq=e, rate_=r), eq_width=eq_width)
        for e, r in rxn_df.select(Reaction.eq, ReactionRate.rate).rows()
    ]
    return block(KeyWord.REACTIONS, rxn_strs)


def block(key, val) -> str:
    """Write a block to a string.

    :param key: The starting key for the block
    :param val: The block value(s)
    :return: The block
    """
    val = val if isinstance(val, str) else "\n".join(val)
    return "\n\n".join([key, val, KeyWord.END])
