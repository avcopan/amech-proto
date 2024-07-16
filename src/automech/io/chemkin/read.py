"""Functions for reading CHEMKIN-formatted files."""

import os
import re

import polars
import pyparsing as pp
from pyparsing import pyparsing_common as ppc

from ... import data, schema
from ..._mech import Mechanism
from ..._mech import from_data as mechanism_from_data
from ...schema import (
    Reaction,
    ReactionDataFrame,
    ReactionRate,
    Species,
    SpeciesDataFrame,
)
from ...util import df_

# generic
COMMENT_REGEX = re.compile(r"!.*$", flags=re.M)
HASH_COMMENT_REGEX = re.compile(r"# .*$", flags=re.M)
COMMENT_START = pp.Suppress(pp.Literal("!"))
COMMENT_END = pp.Suppress(pp.LineEnd())
COMMENT = COMMENT_START + ... + COMMENT_END
COMMENTS = pp.ZeroOrMore(COMMENT)

# units
E_UNIT = pp.Opt(
    pp.CaselessKeyword("CAL/MOLE")
    ^ pp.CaselessKeyword("KCAL/MOLE")
    ^ pp.CaselessKeyword("JOULES/MOLE")
    ^ pp.CaselessKeyword("KJOULES/MOLE")
    ^ pp.CaselessKeyword("KELVINS")
)
A_UNIT = pp.Opt(pp.CaselessKeyword("MOLES") ^ pp.CaselessKeyword("MOLECULES"))

# reactions
SPECIES_NAME = data.reac.SPECIES_NAME
ARROW = data.reac.ARROW
FALLOFF = data.reac.FALLOFF
DUP = pp.Opt(pp.CaselessKeyword("DUP") ^ pp.CaselessKeyword("DUPLICATE"))


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
    return mechanism_from_data(inp=rxn_df, spc_inp=spc_df)


# reactions
def reactions(inp: str, out: str | None = None) -> ReactionDataFrame:
    """Extract reaction information as a dataframe from a CHEMKIN file.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The reactions dataframe
    """
    # Build the parser
    r_expr = pp.Group(
        pp.delimitedList(SPECIES_NAME, delim="+")("species")
        + pp.Opt(FALLOFF)("falloff")
    )
    eq_expr = r_expr("reactants") + ARROW("arrow") + r_expr("products")
    rxn_expr = (
        eq_expr
        + number_list_expr(3)("arrh")
        + pp.Opt(rate_params_expr("LOW", 3))("arrh0")
        + pp.Opt(rate_params_expr("TROE", 3, 4))("troe")
        + pp.Opt(pp.OneOrMore(pp.Group(rate_params_expr("PLOG", 4))))("plog")
        + DUP("dup")
    )
    parser = pp.Suppress(...) + pp.OneOrMore(pp.Group(rxn_expr))

    # Do the parsing
    rxn_block_str = reactions_block(inp, comments=False)
    names = []
    rates = []
    for res in parser.parseString(rxn_block_str):
        rxn = data.reac.from_chemkin(
            rcts=list(res["reactants"]["species"]),
            prds=list(res["products"]["species"]),
            arrow=res["arrow"],
            falloff=res.get("falloff", ""),
            arrh=res.get("arrh", None),
            arrh0=res.get("arrh0", None),
            troe=res.get("arrh0", None),
        )

        names.append(data.reac.equation(rxn))
        rates.append(data.reac.rate(rxn))

    data_dct = {Reaction.eq: names, ReactionRate.rate: rates}
    rxn_df = polars.DataFrame(
        data=data_dct, schema=schema.types([Reaction, ReactionRate])
    )

    rxn_df = schema.reaction_table(rxn_df, models=(Reaction, ReactionRate))
    df_.to_csv(rxn_df, out)

    return rxn_df


def reactions_block(inp: str, comments: bool = True) -> str:
    """Get the reactions block, starting with 'REACTIONS' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, "REACTIONS", comments=comments)


def reactions_units(inp: str, default: bool = True) -> tuple[str, str]:
    """Get the E and A units for reaction rate constants.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param default: Return default values, if missing?
    :return: The units for E and A, respectively
    """
    e_default = "CAL/MOL" if default else None
    a_default = "MOLES" if default else None

    rxn_block_str = reactions_block(inp, comments=False)
    parser = E_UNIT("e_unit") + A_UNIT("a_unit")
    unit_dct = parser.parseString(rxn_block_str).as_dict()
    e_unit = unit_dct["e_unit"].upper() if "e_unit" in unit_dct else e_default
    a_unit = unit_dct["a_unit"].upper() if "a_unit" in unit_dct else a_default
    return e_unit, a_unit


# species
def species(inp: str, out: str | None = None) -> SpeciesDataFrame:
    """Get the list of species, along with their comments.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: A dictionary mapping species onto their comments
    """
    word = pp.Word(pp.printables, exclude_chars=":")
    value = pp.Group(word + pp.Suppress(":") + word)
    values = pp.ZeroOrMore(value)
    comment_values = COMMENT_START + values + COMMENT_END
    entry = SPECIES_NAME("name") + comment_values("values")  # + pp.Suppress(COMMENTS)
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
    return block(inp, "SPECIES", comments=comments)


def species_names(inp: str) -> list[str]:
    """Get the list of species.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The species
    """
    parser = pp.OneOrMore(SPECIES_NAME)
    spc_block_str = species_block(inp, comments=False)
    return parser.parseString(spc_block_str).asList()


# therm
def therm_block(inp: str) -> str:
    """Get the therm block, starting with 'REACTIONS' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, "THERM")


# generic
def block(inp: str, key: str, comments: bool = False) -> str:
    """Get a keyword block, starting with a key and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param key: The key that the block starts with
    :param comments: Include comments?
    :return: The block
    """
    inp = open(inp).read() if os.path.exists(inp) else inp

    block_par = pp.Suppress(...) + pp.QuotedString(
        key, end_quote_char="END", multiline=True
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
    inp = open(inp).read() if os.path.exists(inp) else inp

    inp = re.sub(COMMENT_REGEX, "", inp)
    return re.sub(HASH_COMMENT_REGEX, "", inp)


def all_comments(inp: str) -> list[str]:
    """Get all comments from a CHEMKIN string or substring.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The comments
    """
    inp = open(inp).read() if os.path.exists(inp) else inp

    return re.findall(COMMENT_REGEX, inp)


# helpers
def number_list_expr(
    nmin: int, nmax: int | None = None, delim: str = ""
) -> pp.core.ParseExpression:
    """Get a parse expression for a list of numbers.

    :param nmin: The minimum list length
    :param nmax: The maximum list length (defaults to `nmin` if `None`)
    :param delim: The delimiter between numbers, defaults to ""
    :return: The parse expression
    """
    nmax = nmin if nmax is None else nmax
    return pp.delimitedList(ppc.number, delim=delim, min=nmin, max=nmax)


def rate_params_expr(
    key: str, nmin: int, nmax: int | None = None
) -> pp.core.ParseExpression:
    """Get parse expression for rate parameters after a CHEMKIN reaction.

    :param key: The keyword for these rate parameters
    :param nmin: The minimum parameter list length
    :param nmax: The maximum parameter list length (defaults to `nmin` if `None`)
    :return: The parse expression
    """
    keyword = pp.Suppress(pp.CaselessLiteral(key))
    slash = pp.Suppress(pp.Literal("/"))
    params = number_list_expr(nmin, nmax=nmax, delim="")
    return keyword + slash + params + slash
