"""Functions for reading CHEMKIN-formatted files."""

import itertools
import os
import re
from pathlib import Path

import more_itertools as mit
import polars
import pyparsing as pp

from ... import data, schema
from ..._mech import Mechanism
from ..._mech import from_data as mechanism_from_data
from ...schema import Reaction, ReactionRate, Species
from ...util import df_


class KeyWord:
    # Blocks
    ELEMENTS = "ELEMENTS"
    THERM = "THERM"
    SPECIES = "SPECIES"
    REACTIONS = "REACTIONS"
    END = "END"
    # Units
    # # Energy (E) units
    CAL_MOLE = "CAL/MOLE"
    KCAL_MOLE = "KCAL/MOLE"
    JOULES_MOLE = "JOULES/MOLE"
    KJOULES_MOLE = "KJOULES/MOLE"
    KELVINS = "KELVINS"
    # # Prefactor (A) units
    MOLES = "MOLES"
    MOLECULES = "MOLECULES"


# generic
COMMENT_REGEX = re.compile(r"!.*$", flags=re.M)
HASH_COMMENT_REGEX = re.compile(r"# .*$", flags=re.M)
COMMENT_START = pp.Suppress(pp.Literal("!"))
COMMENT_END = pp.Suppress(pp.LineEnd())
COMMENT = COMMENT_START + ... + COMMENT_END
COMMENTS = pp.ZeroOrMore(COMMENT)

# units
E_UNIT = pp.Opt(
    pp.CaselessKeyword(KeyWord.CAL_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KCAL_MOLE)
    ^ pp.CaselessKeyword(KeyWord.JOULES_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KJOULES_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KELVINS)
)
A_UNIT = pp.Opt(
    pp.CaselessKeyword(KeyWord.MOLES) ^ pp.CaselessKeyword(KeyWord.MOLECULES)
)


# mechanism
def mechanism(
    inp: str, out: str | None = None, spc_out: str | None = None
) -> Mechanism:
    """Extract the mechanism from a CHEMKIN file.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    rxn_df = reactions(inp, out=out)
    spc_df = species(inp, out=spc_out)
    return mechanism_from_data(rxn_inp=rxn_df, spc_inp=spc_df)


# reactions
def reactions(inp: str, out: str | None = None) -> polars.DataFrame:
    """Extract reaction information as a dataframe from a CHEMKIN file.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The reactions dataframe
    """

    def _is_reaction_line(string: str) -> bool:
        return re.search(r"\d$", string.strip())

    # Do the parsing
    rxn_block_str = reactions_block(inp, comments=False)
    line_iter = itertools.dropwhile(
        lambda s: not _is_reaction_line(s), rxn_block_str.splitlines()
    )
    rxn_strs = list(map("\n".join, mit.split_before(line_iter, _is_reaction_line)))

    rxns = list(map(data.reac.from_chemkin_string, rxn_strs))
    eqs = list(map(data.reac.equation, rxns))
    rates = list(map(dict, map(data.reac.rate, rxns)))
    coll_dcts = list(map(data.reac.colliders, rxns))
    # Polars doesn't allow missing values for Struct-valued columns, so replace `None`
    # with an empty collider dictionary
    coll_dcts = [{"M": None} if d is None else d for d in coll_dcts]

    data_dct = {
        Reaction.eq: eqs,
        ReactionRate.rate: rates,
        ReactionRate.colliders: coll_dcts,
    }
    schema_dct = schema.types([Reaction, ReactionRate])
    rxn_df = polars.DataFrame(data=data_dct, schema=schema_dct)

    rxn_df = schema.reaction_table(rxn_df, models=(Reaction, ReactionRate))
    df_.to_csv(rxn_df, out)

    return rxn_df


def reactions_block(inp: str, comments: bool = True) -> str:
    """Get the reactions block, starting with 'REACTIONS' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.REACTIONS, comments=comments)


def reactions_units(inp: str, default: bool = True) -> tuple[str, str]:
    """Get the E and A units for reaction rate constants.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param default: Return default values, if missing?
    :return: The units for E and A, respectively
    """
    e_default = KeyWord.CAL_MOLE if default else None
    a_default = KeyWord.MOLES if default else None

    rxn_block_str = reactions_block(inp, comments=False)
    parser = E_UNIT("e_unit") + A_UNIT("a_unit")
    res = parser.parseString(rxn_block_str).as_dict()
    e_unit = res.get("e_unit", e_default)
    a_unit = res.get("a_unit", a_default)
    e_unit = e_unit.upper() if isinstance(e_unit, str) else None
    a_unit = a_unit.upper() if isinstance(a_unit, str) else None
    return e_unit, a_unit


# species
def species(inp: str, out: str | None = None) -> polars.DataFrame:
    """Get the list of species, along with their comments.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: A dictionary mapping species onto their comments
    """
    species_name = pp.Word(pp.printables)
    word = pp.Word(pp.printables, exclude_chars=":")
    value = pp.Group(word + pp.Suppress(":") + word)
    values = pp.ZeroOrMore(value)
    comment_values = COMMENT_START + values + COMMENT_END
    entry = species_name("name") + comment_values("values")
    parser = pp.Suppress(...) + pp.OneOrMore(pp.Group(entry))

    spc_block_str = species_block(inp, comments=True)

    data_lst = [
        {Species.name: r.get("name"), **dict(r.get("values").asList())}
        for r in parser.parseString(spc_block_str)
    ]
    spc_df = polars.DataFrame(data_lst)

    spc_df = schema.species_table(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df


def species_block(inp: str, comments: bool = True) -> str:
    """Get the species block, starting with 'SPECIES' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.SPECIES, comments=comments)


def species_names(inp: str) -> list[str]:
    """Get the list of species.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The species
    """
    parser = pp.OneOrMore(pp.Word(pp.printables))
    spc_block_str = species_block(inp, comments=False)
    return parser.parseString(spc_block_str).asList()


# therm
def therm_block(inp: str) -> str:
    """Get the therm block, starting with 'REACTIONS' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.THERM)


# generic
def block(inp: str, key: str, comments: bool = False) -> str:
    """Get a keyword block, starting with a key and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param key: The key that the block starts with
    :param comments: Include comments?
    :return: The block
    """
    inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)

    block_par = pp.Suppress(...) + pp.QuotedString(
        key, end_quote_char=KeyWord.END, multiline=True
    )
    (block_str,) = block_par.parseString(inp).asList()
    # Remove comments, if requested
    if not comments:
        block_str = without_comments(block_str)
    return block_str


def without_comments(inp: str) -> str:
    """Get a CHEMKIN string or substring with comments removed.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The string, without comments
    """
    inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)

    inp = re.sub(COMMENT_REGEX, "", inp)
    return re.sub(HASH_COMMENT_REGEX, "", inp)


def all_comments(inp: str) -> list[str]:
    """Get all comments from a CHEMKIN string or substring.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The comments
    """
    inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)

    return re.findall(COMMENT_REGEX, inp)
