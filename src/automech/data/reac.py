"""Reaction dataclasses."""
import dataclasses
import re
import textwrap
from collections import defaultdict

import pyparsing as pp
from pyparsing import pyparsing_common as ppc

from . import rate as rt_
from .rate import BlendType, Rate, RateType


@dataclasses.dataclass
class Reaction:
    """A reaction.

    :param reactants: Names for the reactants
    :param products: Names for the products
    :param rate: The reaction rate
    :param collider: The collider type: 'M', 'He', 'Ne', etc. (currently only 'M')
    """

    reactants: tuple[str, ...]
    products: tuple[str, ...]
    rate: Rate | None = None
    collider: str | None = None

    def __post_init__(self):
        """Initialize attributes."""
        # Set the collider to `None` if there isn't one
        if not rt_.has_collider or not self.collider:
            self.collider = None

        # Set the collider to 'M' if there should be one, but it wasn't specified
        if rt_.has_collider(self.rate) and self.collider is None:
            self.collider = "M"


# getters
def reactants(rxn: Reaction) -> tuple[str, ...]:
    """Get the list of reactants.

    :param rxn: A reaction object
    :return: The CHEMKIN names of the reactants
    """
    return rxn.reactants


def products(rxn: Reaction) -> tuple[str, ...]:
    """Get the list of products.

    :param rxn: A reaction object
    :return: The CHEMKIN names of the products
    """
    return rxn.products


def rate(rxn: Reaction) -> Rate:
    """Get the rate constant.

    :param rxn: A reaction object
    :return: The rate object
    """
    return rxn.rate


def collider(rxn: Reaction) -> str | None:
    """Get the collider, if there is one.

    :param rxn: A reaction object
    :param format: Put a falloff collider in CHEMKIN format?
    :return: The collider
    """
    return rxn.collider


# properties
def is_falloff(rxn: Reaction) -> bool:
    """Whether this is a fallof reaction.

    :param rxn: A reaction object
    :return: `True` if it is, `False` if it isn't
    """
    return rt_.is_falloff(rate(rxn))


def equation(rxn: Reaction) -> str:
    """Get the CHEMKIN equation of a reaction (excludes collider term).

    :param rxn: A reaction object
    :return: The reaction CHEMKIN equation
    """
    return write_chemkin_equation(reactants(rxn), products(rxn))


def chemkin_collider(rxn: Reaction) -> str | None:
    """Get the CHEMKIN collider name for a reaction.

    :param rxn: A reaction object
    :return: The CHEMKIN collider name
    """
    coll = collider(rxn)
    return f"(+{coll})" if is_falloff(rxn) else coll


def chemkin_reagents(
    rxn: Reaction, tuple_coll: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Get the CHEMKIN reactants, products, and collider names for a reaction.

    :param rxn: A reaction object
    :param tuple_coll: Whether to return the collider as a tuple, defaults to False
    :return: The CHEMKIN names of the reactants, products, and collider(s)
    """
    coll = chemkin_collider(rxn)
    return reactants(rxn), products(rxn), (coll,) if tuple_coll else coll


def chemkin_equation(rxn: Reaction) -> str:
    """Get the CHEMKIN equation of a reaction (includes collider term).

    :param rxn: A reaction object
    :return: The reaction CHEMKIN equation
    """
    rcts, prds, coll = chemkin_reagents(rxn)
    return write_chemkin_equation(rcts, prds, coll=coll)


# I/O
def chemkin_string(rxn: Reaction, eq_width: int = 64) -> str:
    """Write a Reaction object to a CHEMKIN string.

    :param rxn: A reaction object
    :param eq_width: The column width for the reaction equation
    :return: The CHEMKIN reaction string
    """
    eq = chemkin_equation(rxn)
    rate_ = rate(rxn)
    rxn_str = f"{eq:<{eq_width}} {write_chemkin_rate(rate_)}"
    print(rxn_str)


def from_chemkin_string(rxn_str: str) -> Reaction:
    """Get a Reaction object from a CHEMKIN string.

    :param rxn_str: CHEMKIN reaction data
    :return: The reaction object
    """
    rcts, prds, coll, arrow = read_chemkin_equation(rxn_str)
    rate = read_chemkin_rate(rxn_str, coll=coll, arrow=arrow)
    return Reaction(reactants=tuple(rcts), products=tuple(prds), rate=rate)


# Chemkin equation helpers
def write_chemkin_equation(
    rcts: tuple[str],
    prds: tuple[str],
    coll: str | None = None,
    arrow: str = "=",
    trans_dct: dict[str, str] | None = None,
) -> str:
    """Write the CHEMKIN equation of a reaction to a string.

    :param rcts: The reactant names
    :param prds: The product names
    :param coll: The collider
    :param trans_dct: Optionally, translate the species names using a dictionary
    :return: The reaction CHEMKIN equation
    """

    def trans_(name):
        return name if trans_dct is None else trans_dct.get(name)

    rcts_ = list(map(trans_, rcts))
    prds_ = list(map(trans_, prds))

    assert all(
        isinstance(n, str) for n in rcts_ + prds_
    ), f"Some species in {rcts}={prds} have no translation:\n{trans_dct}"

    rcts_str = " + ".join(rcts_)
    prds_str = " + ".join(prds_)

    if coll is not None:
        coll = (
            coll if isinstance(coll, str) else coll[0]
        )  # For mechanalyzer compatibility
        sep = " " if "+" in coll else " + "
        rcts_str = sep.join([rcts_str, coll])
        prds_str = sep.join([prds_str, coll])

    return f" {arrow} ".join([rcts_str, prds_str])


def read_chemkin_equation(
    rxn_str: str,
    trans_dct: dict[str, str] | None = None,
    bare_coll: bool = False,
    tuple_coll: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Parse the CHEMKIN equation of a reaction from a string.

    :param rxn_str: CHEMKIN reaction data
    :param trans_dct: Optionally, translate the species names using a dictionary
    :param bare_coll: Return a bare collider, without parentheses?
    :param tuple_coll: Use tuple colliders, for mechanalyzer compatibility? (temporary)
    :return: The reactants and products, along with the colliders
    """

    def trans_(name):
        return name if trans_dct is None else trans_dct.get(name)

    # 0. Find the equation
    rxn_params = number_list_expr(3)
    rxn_eq = pp.Suppress(...) + pp.Combine(
        pp.OneOrMore(pp.Word(pp.printables), stop_on=rxn_params), adjacent=False
    )("eq")
    res = rxn_eq.parseString(rxn_str)
    eq = res.get("eq")

    # 1. Find the arrow and split
    eq_expr = (
        pp.SkipTo(ARROW)("side1") + ARROW("arrow") + pp.SkipTo(pp.StringEnd())("side2")
    )
    res = eq_expr.parseString(eq)
    arrow = res.get("arrow")

    # 2. For each side, find the collider and the reagents
    rcts, falloff = read_chemkin_equation_side(res.get("side1"))
    prds, falloff_ = read_chemkin_equation_side(res.get("side2"))
    assert not (falloff is None) ^ (
        falloff_ is None
    ), f"Inconsistent equation: {rxn_str}"

    rcts, prds, coll = extract_collider(rcts, prds, falloff)
    rcts = tuple(map(trans_, rcts))
    prds = tuple(map(trans_, prds))

    # If requested, remove (+ ) for falloff reactions and return only the bare collider
    if bare_coll:
        if coll is not None and coll.startswith("("):
            coll = coll[1:-1].lstrip(" +")

    if tuple_coll:
        coll = (coll,)

    return (rcts, prds, coll, arrow)


def standardize_chemkin_equation(eq: str) -> str:
    """Standardize the format of a CHEMKIN equation for string comparison.

    :param eq: The reaction CHEMKIN equation
    :return: The reaction CHEMKIN equation in standard format
    """
    return write_chemkin_equation(*read_chemkin_equation(eq))


# Chemkin rate helpers
def write_chemkin_rate(rate_: Rate, indent: int = 4) -> str:
    """Write the CHEMKIN rate to a string.

    :param rate_: The reaction rate object
    :return: CHEMKIN rate string
    """
    # Write the top line
    top_k = rt_.arrhenius_function(rate_)
    top_line = rt_.arrhenius_params_string(top_k)

    # Write the aux lines and indent them
    aux_lines = []

    if rt_.type_(rate_) == RateType.ACTIVATED:
        k_hi = rt_.high_p_arrhenius_function(rate_)
        aux_lines.append(f"HIGH /{rt_.arrhenius_params_string(k_hi)}/")

    if rt_.type_(rate_) == RateType.FALLOFF:
        k_lo = rt_.low_p_arrhenius_function(rate_)
        aux_lines.append(f"LOW  /{rt_.arrhenius_params_string(k_lo)}/")

    if rt_.type_(rate_) == RateType.PLOG:
        ks = rt_.plog_arrhenius_functions(rate_)
        ps = rt_.plog_pressures(rate_)
        aux_lines.extend(
            [
                f"PLOG /{rt_.write_number(p)} {rt_.arrhenius_params_string(k)}/"
                for p, k in zip(ps, ks, strict=True)
            ]
        )

    if rt_.blend_type(rate_) == BlendType.TROE:
        coeffs = rt_.blend_coeffs(rate_)
        aux_lines.append(f"TROE /{rt_.write_numbers(coeffs)}/")

    aux_str = textwrap.indent("\n".join(aux_lines), " " * indent)
    return "\n".join([top_line, aux_str])


def read_chemkin_rate(rate_str: str, coll: str | None = None, arrow: str = "=") -> Rate:
    """Read the CHEMKIN rate from a string.

    :param rxn_str: CHEMKIN rate string
    :return: The reaction rate object
    """
    # Parse the string
    rate_expr = chemkin_rate_expr()
    res = rate_expr.parseString(rate_str).as_dict()

    # Gather auxiliary data
    aux_dct = defaultdict(list)
    for key, *val in res.get("aux"):
        if key == "PLOG":
            aux_dct[key].append(val)
        else:
            aux_dct[key].extend(val)
    aux_dct = dict(aux_dct)

    # Pre-process P-Log data
    plog_ks = plog_ps = None
    if "PLOG" in aux_dct:
        plog_ks = tuple(c[1:] for c in aux_dct["PLOG"])
        plog_ps = tuple(c[0] for c in aux_dct["PLOG"])

    # Pre-process blending function data
    f = None
    if "TROE" in aux_dct:
        f = (BlendType.TROE, aux_dct.get("TROE"))

    # Call central constructor
    return rt_.from_data(
        k=res.get("params"),
        k0=aux_dct.get("LOW"),
        f=f,
        plog_ks=plog_ks,
        plog_ps=plog_ps,
        type_=(
            None
            if coll is None
            else RateType.FALLOFF
            if "(" in coll
            else RateType.ACTIVATED
        ),
        is_rev=arrow in ("=", "<=>"),
    )


# Extra helpers
def number_list_expr(
    nmin: int | None = None, nmax: int | None = None, delim: str = ""
) -> pp.ParseExpression:
    """Get a parse expression for a list of numbers.

    :param nmin: The minimum list length (defaults to `None`)
    :param nmax: The maximum list length (defaults to `nmin` if `None`)
    :param delim: The delimiter between numbers, defaults to ""
    :return: The parse expression
    """
    nmax = nmin if nmax is None else nmax
    return pp.delimitedList(ppc.number, delim=delim, min=nmin, max=nmax)


SLASH = pp.Suppress(pp.Literal("/"))
ARROW = pp.Literal("=") ^ pp.Literal("=>") ^ pp.Literal("<=>")
FALLOFF = pp.Combine(
    pp.Literal("(") + pp.Literal("+") + pp.Word(pp.alphanums) + pp.Literal(")"),
    adjacent=False,
)
ARRH_PARAMS = number_list_expr(3)
AUX_KEYWORD = pp.Word(pp.alphanums)
AUX_PARAMS = SLASH + number_list_expr() + SLASH
AUX_LINE = pp.Group(AUX_KEYWORD + pp.Optional(AUX_PARAMS))
RATE_EXPR = (
    pp.Suppress(...) + ARRH_PARAMS("params") + pp.Group(pp.ZeroOrMore(AUX_LINE))("aux")
)


def chemkin_rate_expr() -> pp.ParseExpression:
    """Get the parse expression for chemkin rates."""
    return RATE_EXPR


def extract_collider(
    rcts: tuple[str, ...],
    prds: tuple[str, ...],
    falloff: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Extract a collider from a list of reactants and products.

    :param rcts: The reactant names
    :param prds: The product names
    :param falloff: A falloff string, if present
    :return: The reactants and products (without collider), and the collider
    """
    colliders = ("M", "He", "Ne", "Ar")

    coll = falloff
    if rcts[-1] == prds[-1] and rcts[-1] in colliders:
        assert falloff is None, f"Collider inconsistency: {falloff} {rcts} {prds}"
        coll = rcts[-1]
        rcts = rcts[:-1]
        prds = prds[:-1]

    return tuple(rcts), tuple(prds), coll


def read_chemkin_equation_side(side: str) -> tuple[list[str], str | None]:
    """Read one side of a CHEMKIN equation.

    :param side: One side of a CHEMKIN equation
    :return: The reagent names and the collider, if any
    """
    # Use pyparsing to split off the falloff value, if present
    end = FALLOFF | pp.StringEnd()
    side_expr = (
        pp.SkipTo(end)("side") + pp.Optional(FALLOFF)("falloff") + pp.StringEnd()
    )
    res = side_expr.parseString(side)
    side = res.get("side")
    coll = res.get("falloff")
    # Use re to split the list of species names
    names = re.split(r"\+(?!\+|$)", side)
    return names, coll


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
