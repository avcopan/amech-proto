"""Reaction dataclasses
"""

import dataclasses
from typing import Dict, Optional, Tuple

import pyparsing as pp

from mecha.data import rate as rate_

# Chemkin parsers
SPECIES_NAME_START = pp.WordStart(pp.alphas)
SPECIES_NAME_BODY = pp.Word(pp.printables, exclude_chars="+=<>!)") + ~pp.FollowedBy("+")
SPECIES_NAME_END = pp.Word(pp.printables, exclude_chars="+=<>!(")

SPECIES_NAME = pp.Combine(SPECIES_NAME_START + pp.Opt(SPECIES_NAME_BODY) + pp.Opt(SPECIES_NAME_END))
ARROW = pp.Literal("=") ^ pp.Literal("=>") ^ pp.Literal("<=>")
FALLOFF = pp.Combine(
    pp.Literal("(") + pp.Literal("+") + pp.Literal("M") + pp.Literal(")"),
    adjacent=False,
)


@dataclasses.dataclass
class Reaction:
    """A reaction

    :param reactants: Names for the reactants
    :param products: Names for the products
    :param rate: The reaction rate
    :param collider: The collider type: 'M', 'He', 'Ne', etc. (currently only 'M')
    """

    reactants: Tuple[str, ...]
    products: Tuple[str, ...]
    rate: Optional[rate_.Rate] = None
    collider: Optional[str] = None

    def __post_init__(self):
        # Set the collider to `None` if there isn't one
        if not rate_.has_collider or not self.collider:
            self.collider = None

        # Set the collider to 'M' if there should be one, but it wasn't specified
        if rate_.has_collider(self.rate) and self.collider is None:
            self.collider = "M"


# constructors
def from_equation(eq: str, rate: rate_.Rate, coll: Optional[str] = None) -> Reaction:
    """Build a Reaction object from an equation string

    :param eq: The CHEMKIN equation
    :param rate: The reaction rate
    :param collider: The collider type
    :return: The reaction object
    """
    rcts, prds, coll_ = read_chemkin_equation(eq, bare_coll=True)
    coll = coll if coll is not None else coll_
    return Reaction(reactants=rcts, products=prds, rate=rate, collider=coll)


def from_chemkin(
    rcts: Tuple[str, ...],
    prds: Tuple[str, ...],
    arrow: str = "=",
    falloff: Optional[str] = None,
    arrh: Optional[rate_.Params3] = None,
    arrh0: Optional[rate_.Params4] = None,
    troe: Optional[rate_.Params3or4] = None,
    plog: Optional[Tuple[rate_.Params4, ...]] = None,
) -> Reaction:
    """Build a Reaction object from CHEMKIN parsing data

    :param rcts: The CHEMKIN reactants
    :param prds: The CHEMKIN products
    :param arrow: The CHEMKIN arrow, indicating whether or not the reaction is reversible
    :param falloff: The CHEMKIN falloff term, '(+M)', if present
    :param arrh: The high-pressure Arrhenius parameters, defaults to None
    :param arrh0: The low-pressure Arrhenius parameters, defaults to None
    :param troe: The Troe parameters, defaults to None
    :param plog: The Plog parameters, defaults to None
    :return: The reaction object
    """
    rcts, prds, coll = extract_collider(rcts, prds)
    if falloff is not None:
        assert coll is None, f"Cannot have collider {coll} with falloff {falloff}"
        coll = falloff

    rate = rate_.from_chemkin(
        arrow=arrow, coll=coll, arrh=arrh, arrh0=arrh0, troe=troe, plog=plog
    )
    return Reaction(reactants=tuple(rcts), products=tuple(prds), rate=rate)


# getters
def reactants(rxn: Reaction) -> Tuple[str, ...]:
    """The list of reactants

    :param rxn: A reaction object
    :return: The CHEMKIN names of the reactants
    """
    return rxn.reactants


def products(rxn: Reaction) -> Tuple[str, ...]:
    """The list of products

    :param rxn: A reaction object
    :return: The CHEMKIN names of the products
    """
    return rxn.products


def rate(rxn: Reaction) -> rate_.Rate:
    """The rate constant or, if multiple, the (arbitrary) first one in the list

    :param rxn: A reaction object
    :return: The rate object
    """
    return rxn.rate


def collider(rxn: Reaction) -> Optional[str]:
    """The collider, if there is one

    :param rxn: A reaction object
    :param format: Put a falloff collider in CHEMKIN format?
    :return: The collider
    """
    return rxn.collider


# properties
def is_falloff(rxn: Reaction) -> bool:
    """Is this a falloff reaction?

    :param rxn: A reaction object
    :return: `True` if it is, `False` if it isn't
    """
    return rate_.is_falloff(rate(rxn))


def equation(rxn: Reaction) -> str:
    """Get the CHEMKIN equation of a reaction (excludes collider term)

    :param rxn: A reaction object
    :return: The reaction CHEMKIN equation
    """
    return write_chemkin_equation(reactants(rxn), products(rxn))


def chemkin_collider(rxn: Reaction) -> Optional[str]:
    coll = collider(rxn)
    return f"(+{coll})" if is_falloff(rxn) else coll


def chemkin_reagents(
    rxn: Reaction, tuple_coll: bool = False
) -> Tuple[Tuple[str, ...], Tuple[str, ...], str]:
    """Get the CHEMKIN reactants, products, and collider for a reaction

    :param rxn: _description_
    :param tuple_coll: _description_, defaults to False
    :return: _description_
    """
    coll = chemkin_collider(rxn)
    return reactants(rxn), products(rxn), (coll,) if tuple_coll else coll


def chemkin_equation(rxn: Reaction) -> str:
    """Get the CHEMKIN equation of a reaction (includes collider term)

    :param rxn: A reaction object
    :return: The reaction CHEMKIN equation
    """
    rcts, prds, coll = chemkin_reagents(rxn)
    return write_chemkin_equation(rcts, prds, coll=coll)


# Chemkin helpers
def read_chemkin_equation(
    eq: str,
    trans_dct: Optional[Dict[str, str]] = None,
    bare_coll: bool = False,
    tuple_coll: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], str]:
    """Parse the CHEMKIN equation of a reaction into reactants and products

    :param eq: The reaction CHEMKIN equation
    :param trans_dct: Optionally, translate the species names using a dictionary
    :param bare_coll: Return a bare collider, without parentheses?
    :param tuple_coll: Use tuple colliders, for mechanalyzer compatibility? (temporary)
    :return: The reactants and products, along with the
    """

    def trans_(name):
        return name if trans_dct is None else trans_dct.get(name)

    r_expr = pp.Group(
        pp.delimitedList(SPECIES_NAME, delim="+")("species") + pp.Opt(FALLOFF)("falloff")
    )
    parser = r_expr("reactants") + ARROW("arrow") + r_expr("products")
    dct = parser.parseString(eq).asDict()
    rcts = tuple(map(trans_, dct["reactants"]["species"]))
    prds = tuple(map(trans_, dct["products"]["species"]))

    rcts, prds, coll = extract_collider(rcts, prds)
    if "falloff" in dct["reactants"]:
        falloff = dct["reactants"]["falloff"]
        assert "falloff" in dct["products"], f"Failed to parse falloff: {eq}"
        assert coll is None, f"Cannot have collider {coll} with falloff {falloff}"

        coll = falloff.replace(" ", "")

    # If requested, remove (+ ) for falloff reactions and return only the bare collider
    if bare_coll:
        if coll is not None and coll.startswith("("):
            coll = coll[1:-1].lstrip(" +")

    if tuple_coll:
        coll = (coll,)

    return (rcts, prds, coll)


def write_chemkin_equation(
    rcts: Tuple[str],
    prds: Tuple[str],
    coll: Optional[str] = None,
    trans_dct: Optional[Dict[str, str]] = None,
) -> str:
    """Form the CHEMKIN equation of a reaction from reactants and products

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

    if not all(isinstance(n, str) for n in rcts_ + prds_):
        print(f"Some species in {rcts}={prds} have no translation:\n{trans_dct}")
        return None

    rcts_str = " + ".join(rcts_)
    prds_str = " + ".join(prds_)

    if coll is not None:
        coll = (
            coll if isinstance(coll, str) else coll[0]
        )  # For mechanalyzer compatibility
        sep = " " if "+" in coll else " + "
        rcts_str = sep.join([rcts_str, coll])
        prds_str = sep.join([prds_str, coll])

    return " = ".join([rcts_str, prds_str])


def extract_collider(
    rcts: Tuple[str, ...], prds: Tuple[str, ...]
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    """Extract a collider from a list of reactants and products

    :param rcts: The reactant names
    :param prds: The product names
    :return: The reactants and products (without collider), and the collider
    """
    colliders = ("M", "He", "Ne", "Ar")

    coll = None
    if rcts[-1] == prds[-1] and rcts[-1] in colliders:
        coll = rcts[-1]
        rcts = rcts[:-1]
        prds = prds[:-1]

    return tuple(rcts), tuple(prds), coll
