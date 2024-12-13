"""Definition and core functionality of the mechanism data structure."""

import dataclasses
import itertools
import json
import textwrap
from collections.abc import Collection, Sequence

import automol
import more_itertools as mit
import networkx
import polars

from . import data, reac_table, schema
from . import old_net as net_
from .schema import (
    Model,
    Reaction,
    ReactionMisc,
    ReactionRate,
    ReactionRenamed,
    ReactionStereo,
    Species,
    SpeciesRenamed,
    SpeciesStereo,
    SpeciesThermo,
)
from .util import df_

DEFAULT_EXCLUDE_FORMULAS = ("H*O*", "CH*")


@dataclasses.dataclass
class Mechanism:
    """A chemical kinetic mechanism."""

    reactions: polars.DataFrame
    species: polars.DataFrame
    rate_units: tuple[str, str] | None = None
    thermo_temps: tuple[float, float, float] | None = None

    def __post_init__(self):
        """Initialize attributes."""
        if self.thermo_temps is not None:
            assert len(self.thermo_temps) == 3, f"Bad thermo_temps: {self.thermo_temps}"
            self.thermo_temps = tuple(map(float, self.thermo_temps))

        if self.rate_units is not None:
            assert len(self.rate_units) == 2, f"Bad rate_units: {self.rate_units}"
            self.rate_units = tuple(map(str, self.rate_units))

        if not isinstance(self.reactions, polars.DataFrame):
            self.reactions = polars.DataFrame(self.reactions)

        if not isinstance(self.species, polars.DataFrame):
            self.species = polars.DataFrame(self.species)

        self.species = schema.species_table(self.species)
        self.reactions, _ = schema.reaction_table(self.reactions, spc_df=self.species)

    def __repr__(self):
        rxn_df_rep = textwrap.indent(repr(self.reactions), "  ")
        spc_df_rep = textwrap.indent(repr(self.species), "  ")
        attrib_strs = [
            f"reactions=DataFrame(\n{rxn_df_rep}\n)",
            f"species=DataFrame(\n{spc_df_rep}\n)",
            f"rate_units={self.rate_units}",
            f"thermo_temps={self.thermo_temps}",
        ]
        attrib_strs = [textwrap.indent(s, "  ") for s in attrib_strs]
        attrib_str = ",\n".join(attrib_strs)
        return f"Mechanism(\n{attrib_str},\n)"

    def __iter__(self):
        """For overloading dict conversion."""
        rxn_dct = json.loads(self.reactions.write_json())
        spc_dct = json.loads(self.species.write_json())
        mech_dct = {**self.__dict__, "reactions": rxn_dct, "species": spc_dct}
        yield from mech_dct.items()


# constructors
def from_data(
    rxn_inp: str | polars.DataFrame,
    spc_inp: str | polars.DataFrame,
    thermo_temps: tuple[float, float, float] | None = None,
    rate_units: tuple[str, str] | None = None,
    rxn_models: Sequence[Model] = (),
    spc_models: Sequence[Model] = (),
    fail_on_error: bool = True,
) -> Mechanism:
    """Contruct a mechanism object from data.

    :param rxn_inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param rxn_models: Extra reaction models to validate against
    :param spc_models: Extra species models to validate against
    :param fail_on_error: Whether to raise an exception of there is an inconsistency
    :return: The mechanism object
    """
    spc_df = spc_inp if isinstance(spc_inp, polars.DataFrame) else df_.from_csv(spc_inp)
    rxn_df = rxn_inp if isinstance(rxn_inp, polars.DataFrame) else df_.from_csv(rxn_inp)
    spc_df = schema.species_table(spc_df, models=spc_models)
    rxn_df, _ = schema.reaction_table(
        rxn_df, models=rxn_models, spc_df=spc_df, fail_on_error=fail_on_error
    )
    mech = Mechanism(
        reactions=rxn_df,
        species=spc_df,
        thermo_temps=thermo_temps,
        rate_units=rate_units,
    )
    return mech


def from_network(net: networkx.MultiGraph) -> Mechanism:
    """Generate a mechanism from a reaction network.

    :param net: A reaction network
    :return: The mechanism
    """
    spc_data = [d for *_, d in net.nodes.data()]
    spc_data.extend([d for *_, d in net.graph.get(net_.Key.excluded_species)])
    rxn_data = [d for *_, d in net.edges.data()]
    rxn_data.extend([d for *_, d in net.graph.get(net_.Key.excluded_reactions)])

    spc_df = (
        polars.DataFrame([])
        if not spc_data
        else (
            polars.DataFrame(spc_data)
            .sort(net_.Key.id)
            .unique(net_.Key.id, maintain_order=True)
        )
    )
    rxn_df = (
        polars.DataFrame([])
        if not rxn_data
        else (
            polars.DataFrame(rxn_data)
            .sort(net_.Key.id)
            .unique(net_.Key.id, maintain_order=True)
        )
    )
    return from_data(rxn_inp=rxn_df, spc_inp=spc_df, fail_on_error=False)


def from_smiles(
    spc_smis: Sequence[str] = (),
    rxn_smis: Sequence[str] = (),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
) -> Mechanism:
    """Generate a mechanism, using SMILES strings for the species names.

    If `name_dct` is `None`, CHEMKIN names will be auto-generated.

    :param spc_smis: The species SMILES strings
    :param rxn_smis: Optionally, the reaction SMILES strings
    :param name_dct: Optionally, specify the name for some molecules
    :param spin_dct: Optionally, specify the spin state (2S) for some molecules
    :param charge_dct: Optionally, specify the charge for some molecules
    :return: The mechanism
    """
    name_dct = {} if name_dct is None else name_dct
    spin_dct = {} if spin_dct is None else spin_dct
    charge_dct = {} if charge_dct is None else charge_dct

    # Add in any missing species from the reaction SMILES
    spc_smis_by_rxn = [
        rs + ps
        for (rs, ps) in map(automol.smiles.reaction_reactants_and_products, rxn_smis)
    ]
    spc_smis = list(mit.unique_everseen(itertools.chain(spc_smis, *spc_smis_by_rxn)))

    # Build the species dataframe
    chis = list(map(automol.smiles.amchi, spc_smis))
    chi_dct = dict(zip(spc_smis, chis, strict=True))
    name_dct = {chi_dct[k]: v for k, v in name_dct.items() if k in spc_smis}
    spin_dct = {chi_dct[k]: v for k, v in spin_dct.items() if k in spc_smis}
    charge_dct = {chi_dct[k]: v for k, v in charge_dct.items() if k in spc_smis}
    data_dct = {Species.smiles: spc_smis, Species.amchi: chis}
    dt = schema.species_types(data_dct.keys())
    spc_df = polars.DataFrame(data=data_dct, schema=dt)
    spc_df = schema.species_table(
        spc_df, name_dct=name_dct, spin_dct=spin_dct, charge_dct=charge_dct
    )

    # Build the reactions dataframe
    trans_dct = df_.lookup_dict(spc_df, Species.smiles, Species.name)
    rxn_smis_lst = list(map(automol.smiles.reaction_reactants_and_products, rxn_smis))
    data_lst = [
        {
            Reaction.reactants: list(map(trans_dct.get, rs)),
            Reaction.products: list(map(trans_dct.get, ps)),
        }
        for rs, ps in rxn_smis_lst
    ]
    dt = schema.reaction_types([Reaction.reactants, Reaction.products])
    rxn_df = polars.DataFrame(data=data_lst, schema=dt)
    return from_data(rxn_df, spc_df)


# getters
def species(mech: Mechanism) -> polars.DataFrame:
    """Get the species dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism's species dataframe
    """
    return mech.species


def reactions(mech: Mechanism) -> polars.DataFrame:
    """Get the reactions dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism's reactions dataframe
    """
    return mech.reactions


def thermo_temperatures(mech: Mechanism) -> tuple[float, float, float] | None:
    """Get the thermo temperatures for a mechanism.

    :param mech: The mechanism
    :return: The thermo temperatures
    """
    return mech.thermo_temps


def rate_units(mech: Mechanism) -> tuple[str, str] | None:
    """Get the rate units for a mechanism.

    :param mech: The mechanism
    :return: The rate units
    """
    return mech.rate_units


# getters
def set_species(mech: Mechanism, spc_df: polars.DataFrame) -> Mechanism:
    """Set the species dataframe for a mechanism.

    :param mech: The mechanism
    :param spc_df: The new species dataframe
    :return: The mechanism with updated species
    """
    return from_data(
        rxn_inp=reactions(mech),
        spc_inp=spc_df,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )


def set_reactions(mech: Mechanism, rxn_df: polars.DataFrame) -> Mechanism:
    """Set the reactions dataframe for a mechanism.

    :param mech: The mechanism
    :param rxn_df: The new reactions dataframe
    :return: The mechanism with updated reactions
    """
    return from_data(
        rxn_inp=rxn_df,
        spc_inp=species(mech),
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )


def set_thermo_temperatures(
    mech: Mechanism, temps: tuple[float, float, float] | None
) -> Mechanism:
    """Set the thermo temperatures for a mechanism.

    :param mech: The mechanism
    :param thermo_temps: The new thermo temperatures
    :return: The thermo temperatures
    """
    return from_data(
        rxn_inp=reactions(mech),
        spc_inp=species(mech),
        thermo_temps=temps,
        rate_units=rate_units(mech),
    )


def set_rate_units(
    mech: Mechanism, units: tuple[str, str] | None, scale_rates: bool = True
) -> Mechanism:
    """Get the rate units for a mechanism.

    :param mech: The mechanism
    :param units: The new rate units
    :param scale_rates: Scale the rates if changing units?
    :return: The rate units
    """
    rxn_df = reactions(mech)
    units0 = rate_units(mech)
    if scale_rates and units0 is not None:
        e_unit0, a_unit0 = map(str.lower, units0)
        e_unit, a_unit = map(str.lower, units)
        assert (
            a_unit == a_unit0
        ), f"{a_unit} != {a_unit0} (A conversion not yet implemented)"

        def _convert(rate_dct):
            rate_obj = data.rate.from_data(**rate_dct)
            rate_obj = data.rate.convert_energy_units(rate_obj, e_unit0, e_unit)
            return dict(rate_obj)

        rxn_df = df_.map_(rxn_df, ReactionRate.rate, ReactionRate.rate, _convert)

    return from_data(
        rxn_inp=rxn_df,
        spc_inp=species(mech),
        thermo_temps=thermo_temperatures(mech),
        rate_units=units,
    )


# properties
def species_count(mech: Mechanism) -> int:
    """Get the number of species in a mechanism.

    :param mech: The mechanism
    :return: The number of species
    """
    return species(mech).select(polars.len()).item()


def reaction_count(mech: Mechanism) -> int:
    """Get the number of reactions in a mechanism.

    :param mech: The mechanism
    :return: The number of reactions
    """
    return reactions(mech).select(polars.len()).item()


def species_names(
    mech: Mechanism,
    rxn_only: bool = False,
    formulas: Sequence[str] | None = None,
    exclude_formulas: Sequence[str] = (),
) -> list[str]:
    """Get the names of species in the mechanism.

    :param mech: A mechanism
    :param rxn_only: Only include species that are involved in reactions?
    :param formulas: Formula strings of species to include, using * for wildcard
        stoichiometry
    :param exclude_formulas: Formula strings of species to exclude, using * for wildcard
        stoichiometry
    :return: The species names
    """

    def _formula_matcher(fml_strs):
        """Determine whether a species is excluded."""
        fmls = list(map(automol.form.from_string, fml_strs))

        def _matches_formula(chi):
            fml = automol.amchi.formula(chi)
            return any(automol.form.match(fml, e) for e in fmls)

        return _matches_formula

    spc_df = species(mech)

    if formulas is not None:
        spc_df = df_.map_(
            spc_df, Species.amchi, "match", _formula_matcher(formulas), dtype_=bool
        )
        spc_df = spc_df.filter(polars.col("match"))

    if exclude_formulas:
        spc_df = df_.map_(
            spc_df, Species.amchi, "match", _formula_matcher(exclude_formulas)
        )
        spc_df = spc_df.filter(~polars.col("match"))

    spc_names = spc_df[Species.name].to_list()

    if rxn_only:
        rxn_spc_names = reaction_species_names(mech)
        spc_names = [n for n in spc_names if n in rxn_spc_names]

    return spc_names


def reaction_reactants(mech: Mechanism) -> list[list[str]]:
    """Get the reactants of reactions in the mechanism.

    :param mech: A mechanism
    :return: The reaction reactants
    """
    rxn_df = reactions(mech)
    return rxn_df[Reaction.reactants].to_list()


def reaction_products(mech: Mechanism) -> list[list[str]]:
    """Get the products of reactions in the mechanism.

    :param mech: A mechanism
    :return: The reaction products
    """
    rxn_df = reactions(mech)
    return rxn_df[Reaction.products].to_list()


def reaction_reactants_and_products(
    mech: Mechanism,
) -> list[tuple[list[str], list[str]]]:
    """Get the products of reactions in the mechanism.

    :param mech: A mechanism
    :return: The reaction products
    """
    rxn_df = reactions(mech)
    return rxn_df[[Reaction.reactants, Reaction.products]].rows()


def reaction_equations(mech: Mechanism) -> list[str]:
    """Get the equations of reactions in the mechanism.

    :param mech: A mechanism
    :return: The reaction equations
    """
    rps = reaction_reactants_and_products(mech)
    return list(itertools.starmap(data.reac.write_chemkin_equation, rps))


def reaction_species_names(mech: Mechanism) -> list[str]:
    """Get the names of all species that participate in reactions.

    :param mech: A mechanism
    :return: The reaction species
    """
    eqs = reaction_equations(mech)
    rxn_names = [r + p for r, p, *_ in map(data.reac.read_chemkin_equation, eqs)]
    return list(mit.unique_everseen(itertools.chain(*rxn_names)))


def rename_dict(mech1: Mechanism, mech2: Mechanism) -> tuple[dict[str, str], list[str]]:
    """Generate a dictionary for renaming species names from one mechanism to another.

    :param mech1: A mechanism with the original names
    :param mech2: A mechanism with the desired names
    :return: The dictionary mapping names from `mech1` to those in `mech2`, and a list
        of names from `mech1` that are missing in `mech2`
    """
    match_cols = [Species.amchi, Species.spin, Species.charge]

    # Read in species and names
    spc1_df = species(mech1)
    spc1_df = spc1_df.rename({Species.name: SpeciesRenamed.orig_name})

    spc2_df = species(mech2)
    spc2_df = spc2_df.select([Species.name, *match_cols])

    # Get the names from the first mechanism that are included/excluded in the second
    incl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="inner")
    excl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="anti")

    name_dct = df_.lookup_dict(incl_spc_df, SpeciesRenamed.orig_name, Species.name)
    missing_names = excl_spc_df[SpeciesRenamed.orig_name].to_list()
    return name_dct, missing_names


def reaction_network(
    mech: Mechanism,
) -> networkx.MultiGraph:
    """Generate a network graph representation of the mechanism.

    :param mech: A mechanism
    :param node_exclude_formulas: Formulas for species to be excluded as nodes
    :return: The reaction network
    """
    spc_df = species(mech)
    rxn_df = reactions(mech)

    # Double-check that the reagents are sorted
    rxn_df = schema.reaction_table_with_sorted_reagents(rxn_df)

    # Add species and reaction indices
    spc_df = df_.with_index(spc_df, net_.Key.id)
    rxn_df = df_.with_index(rxn_df, net_.Key.id)

    # Get a dataframe of reagents
    rgt_col = "reagents"
    rgt_exprs = [
        rxn_df.select(polars.col(Reaction.reactants).alias(rgt_col), Reaction.formula),
        rxn_df.select(polars.col(Reaction.products).alias(rgt_col), Reaction.formula),
    ]
    rgt_df = polars.concat(rgt_exprs).group_by(rgt_col).first()

    # Append species data to the reagents dataframe
    names = spc_df[Species.name]
    datas = spc_df.to_struct()
    expr = polars.element().replace_strict(names, datas)
    rgt_df = rgt_df.with_columns(
        polars.col(rgt_col).list.eval(expr).alias(net_.Key.species)
    )

    # Build the network object
    def _node_data_from_dict(dct: dict[str, object]):
        key = tuple(dct.get(rgt_col))
        return (key, dct)

    def _edge_data_from_dict(dct: dict[str, object]):
        key1 = tuple(dct.get(Reaction.reactants))
        key2 = tuple(dct.get(Reaction.products))
        return (key1, key2, dct)

    mech_net = networkx.MultiGraph()
    mech_net.add_nodes_from(map(_node_data_from_dict, rgt_df.to_dicts()))
    mech_net.add_edges_from(map(_edge_data_from_dict, rxn_df.to_dicts()))
    return mech_net


def species_network(
    mech: Mechanism, node_exclude_formulas: Sequence[str] = DEFAULT_EXCLUDE_FORMULAS
) -> networkx.MultiGraph:
    """Generate a network graph representation of the mechanism.

    :param mech: A mechanism
    :param node_exclude_formulas: Formulas for species to be excluded as nodes
    :return: The reaction network
    """
    excl_spc_names = species_names(mech, formulas=node_exclude_formulas)

    def _node_data_from_dicts(dcts: Sequence[dict]) -> dict:
        names = [d.get(Species.name) for d in dcts]
        return list(zip(names, dcts, strict=True))

    def _edge_data_from_dicts(dcts: Sequence[dict], filter: bool = False) -> dict:
        edge_data = []
        for dct in dcts:
            rcts = dct.get(Reaction.reactants)
            prds = dct.get(Reaction.products)
            for edge_key in itertools.product(rcts, prds):
                if not filter or not any(n in excl_spc_names for n in edge_key):
                    edge_data.append((*edge_key, dct))
        return edge_data

    # Prepare node data
    spc_df = species(mech)
    spc_df = df_.with_index(spc_df, net_.Key.id)  # Add IDs for back conversion
    spc_expr = polars.col(Species.name).is_in(excl_spc_names)
    excl_spc_df = spc_df.filter(spc_expr)
    incl_spc_df = spc_df.filter(~spc_expr)

    incl_spc_data = _node_data_from_dicts(incl_spc_df.to_dicts())
    excl_spc_data = _node_data_from_dicts(excl_spc_df.to_dicts())

    # Prepare edge data
    def node_is_excluded_expression(key: str) -> polars.Expr:
        return (
            polars.col(key).list.eval(polars.element().is_in(excl_spc_names)).list.all()
        )

    rxn_df = reactions(mech)
    rxn_df = df_.with_index(rxn_df, net_.Key.id)  # Add IDs for back conversion
    is_excl_expr = node_is_excluded_expression(
        Reaction.reactants
    ) | node_is_excluded_expression(Reaction.products)
    excl_rxn_df = rxn_df.filter(is_excl_expr)
    incl_rxn_df = rxn_df.filter(~is_excl_expr)

    incl_rxn_data = _edge_data_from_dicts(incl_rxn_df.to_dicts(), filter=True)
    excl_rxn_data = _edge_data_from_dicts(excl_rxn_df.to_dicts())

    excl_data = {
        net_.Key.excluded_species: excl_spc_data,
        net_.Key.excluded_reactions: excl_rxn_data,
    }
    mech_net = networkx.MultiGraph(**excl_data)
    mech_net.add_nodes_from(incl_spc_data)
    mech_net.add_edges_from(incl_rxn_data)
    return mech_net


# transformations
def rename(
    mech: Mechanism, name_dct: dict[str, str], drop_missing: bool = False
) -> Mechanism:
    """Rename the species in a mechanism.

    :param mech: A mechanism
    :param name_dct: A dictionary mapping current species names to new species names
    :param drop_missing: Drop missing species from the mechanism?
        Otherwise, they are retained with their original names
    :return: The mechanism with updated species names
    """
    if drop_missing:
        mech = with_species(mech, list(name_dct), strict=drop_missing)

    spc_df = species(mech)
    spc_df = spc_df.with_columns(polars.col(Species.name).replace(name_dct))

    rxn_df = reactions(mech)
    rxn_df = rxn_df.with_columns(
        polars.col(Reaction.reactants).alias(ReactionRenamed.orig_reactants),
        polars.col(Reaction.products).alias(ReactionRenamed.orig_products),
    )

    repl_expr = polars.element().replace(name_dct)
    rxn_df = rxn_df.with_columns(
        polars.col(Reaction.reactants).list.eval(repl_expr),
        polars.col(Reaction.products).list.eval(repl_expr),
    )
    return from_data(
        rxn_inp=rxn_df,
        spc_inp=spc_df,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )


def remove_all_reactions(mech: Mechanism) -> Mechanism:
    """Clear the reactions from a mechanism.

    :param mech: A mechanism
    :return: The mechanism, without reactions
    """
    return set_reactions(mech, reactions(mech).clear())


def add_reactions(mech: Mechanism, rxn_df: polars.DataFrame) -> Mechanism:
    """Add reactions from a DataFrame to a mechanism.

    :param mech: A mechanism
    :param rxn_df: A reactions dataframe
    :return: The mechanism, with added reactions
    """
    rxn_df0 = reactions(mech)
    return set_reactions(mech, polars.concat([rxn_df0, rxn_df], how="diagonal_relaxed"))


def neighborhood(
    mech: Mechanism,
    spc_names: Sequence[str],
    radius: int = 1,
    exclude_formulas: Sequence[str] = DEFAULT_EXCLUDE_FORMULAS,
) -> Mechanism:
    """Determine the neighborhood of a set of species.

    :param mech: A mechanism
    :param spc_names: A list of species names
    :param radius: Maximum distance of neighbors to include
    :param exclude_formulas: Formula strings of molecules to exclude from the network,
        using * for wildcard stoichiometry, defaults to ("H*", "OH*", "O2H*", "CH*")
    :return: The nth neighborhood mechanism
    """
    mech0 = mech
    for _ in range(radius):
        mech = with_species(mech0, spc_names=spc_names, strict=False)
        spc_names = species_names(
            mech, rxn_only=True, exclude_formulas=exclude_formulas
        )
    return mech


def with_species(
    mech: Mechanism, spc_names: Sequence[str] = (), strict: bool = False
) -> Mechanism:
    """Extract a submechanism including species names from a list.

    :param mech: The mechanism
    :param spc_names: The names of the species to be included
    :param strict: Strictly include these species and no others?
    :return: The submechanism
    """
    return _with_or_without_species(
        mech=mech, spc_names=spc_names, without=False, strict=strict
    )


def without_species(mech: Mechanism, spc_names: Sequence[str] = ()) -> Mechanism:
    """Extract a submechanism excluding species names from a list.

    :param mech: The mechanism
    :param spc_names: The names of the species to be included
    :return: The submechanism
    """
    return _with_or_without_species(mech=mech, spc_names=spc_names, without=True)


def _with_or_without_species(
    mech: Mechanism,
    spc_names: Sequence[str] = (),
    without: bool = False,
    strict: bool = False,
) -> Mechanism:
    """Extract a submechanism containing or excluding species names from a list.

    :param mech: The mechanism
    :param spc_names: The names of the species to be included or excluded
    :param without: Extract the submechanism *without* these species?
    :param strict: Strictly include these species and no others?
    :return: The submechanism
    """
    # Build the appropriate filtering expression
    expr = (
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(spc_names))
        .list
    )
    expr = expr.all() if strict else expr.any()
    expr = expr.not_() if without else expr

    rxn_df = reactions(mech)

    rxn_df = rxn_df.filter(expr)
    return without_unused_species(
        from_data(
            rxn_inp=rxn_df,
            spc_inp=species(mech),
            thermo_temps=thermo_temperatures(mech),
            rate_units=rate_units(mech),
        )
    )


def without_unused_species(mech: Mechanism) -> Mechanism:
    """Remove unused species from a mechanism.

    :param mech: The mechanism
    :return: The mechanism, without unused species
    """
    spc_df = species(mech)
    used_names = species_names(mech, rxn_only=True)
    spc_df = spc_df.filter(polars.col(Species.name).is_in(used_names))
    return set_species(mech, spc_df)


def without_duplicate_reactions(mech: Mechanism) -> Mechanism:
    """Remove duplicate reactions from a mechanism.

    :param mech: A mechanism
    :return: The mechanism, without duplicate reactions
    """
    col_tmp = df_.temp_column()
    rxn_df = reactions(mech)
    rxn_df = reac_table.with_reaction_key(rxn_df, col_name=col_tmp)
    rxn_df = rxn_df.unique(col_tmp, maintain_order=True)
    rxn_df = rxn_df.drop(col_tmp)
    return set_reactions(mech, rxn_df)


def expand_stereo(
    mech: Mechanism,
    enant: bool = True,
    strained: bool = False,
) -> tuple[Mechanism, Mechanism]:
    """Expand stereochemistry for a mechanism.

    :param mech: The mechanism
    :param enant: Distinguish between enantiomers?, defaults to True
    :param strained: Include strained stereoisomers?
    :param drop_unused: Drop unused species from the mechanism?
    :return: A mechanism with the classified reactions, and one with the unclassified
    """
    # Read in the mechanism data
    spc_df0: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    # Do the species expansion
    spc_df = _expand_species_stereo(spc_df0, enant=enant, strained=strained)

    if not reaction_count(mech):
        mech = set_species(mech, spc_df)
        return mech, mech

    # Add reactant and product AMChIs
    rxn_df = reac_table.translate_reagents(
        rxn_df,
        trans=spc_df0[Species.name],
        trans_into=spc_df0[Species.amchi],
        rcol_out="ramchis",
        pcol_out="pamchis",
    )

    # Add "orig" prefix to current reactant and product columns
    col_dct = {
        Reaction.reactants: ReactionStereo.orig_reactants,
        Reaction.products: ReactionStereo.orig_products,
    }
    rxn_df = rxn_df.rename(col_dct)

    # Define the expansion function
    name_dct: dict = df_.lookup_dict(
        spc_df, (SpeciesStereo.orig_name, Species.amchi), Species.name
    )

    def _expand_reaction(rchi0s, pchi0s, rname0s, pname0s):
        """Classify a reaction and return the reaction objects."""
        objs = automol.reac.from_amchis(rchi0s, pchi0s, stereo=False)
        rnames_lst = []
        pnames_lst = []
        ts_amchis = []
        for obj in objs:
            sobjs = automol.reac.expand_stereo(obj, enant=enant, strained=strained)
            for sobj in sobjs:
                # Determine the AMChI
                ts_amchi = automol.reac.ts_amchi(sobj)
                # Determine the updated equation
                rchis, pchis = automol.reac.amchis(sobj)
                rnames = tuple(map(name_dct.get, zip(rname0s, rchis, strict=True)))
                pnames = tuple(map(name_dct.get, zip(pname0s, pchis, strict=True)))
                if not all(isinstance(n, str) for n in rnames + pnames):
                    return ([], [])

                rnames_lst.append(rnames)
                pnames_lst.append(pnames)
                ts_amchis.append(ts_amchi)
        return rnames_lst, pnames_lst, ts_amchis

    # Do the expansion
    cols_in = (
        "ramchis",
        "pamchis",
        ReactionStereo.orig_reactants,
        ReactionStereo.orig_products,
    )
    cols_out = (Reaction.reactants, Reaction.products, ReactionStereo.amchi)
    rxn_df = df_.map_(rxn_df, cols_in, cols_out, _expand_reaction)

    # Separate out the error cases
    err_df = rxn_df.filter(polars.col(Reaction.reactants).list.len() == 0)
    rxn_df = rxn_df.filter(polars.col(Reaction.reactants).list.len() != 0)

    # Expand the table by stereoisomers
    err_df = err_df.drop(ReactionStereo.amchi, *col_dct.keys()).rename(
        dict(map(reversed, col_dct.items()))
    )
    rxn_df = rxn_df.explode(Reaction.reactants, Reaction.products, ReactionStereo.amchi)

    # Form the new mechanisms
    mech = from_data(
        rxn_df,
        spc_df,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )
    err_mech = from_data(
        err_df,
        spc_df0,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )

    return mech, err_mech


def _expand_species_stereo(
    spc_df: polars.DataFrame, enant: bool = True, strained: bool = False
) -> polars.DataFrame:
    """Stereoexpand the species from a mechanism.

    :param spc_df: A species table, as a dataframe
    :param enant: Distinguish between enantiomers?
    :param strained: Include strained stereoisomers?
    :return: The stereoexpanded species table
    """

    # Do the species expansion based on AMChIs
    def _expand_amchi(chi):
        """Expand stereo for the AMChIs."""
        return automol.amchi.expand_stereo(chi, enant=enant)

    spc_df = spc_df.rename({Species.amchi: SpeciesStereo.orig_amchi})
    spc_df = df_.map_(spc_df, SpeciesStereo.orig_amchi, Species.amchi, _expand_amchi)
    spc_df = spc_df.explode(polars.col(Species.amchi))

    # Update the species names
    def _stereo_name(orig_name, chi):
        """Determine the stereo name from the AMChI."""
        return automol.amchi.chemkin_name(chi, root_name=orig_name)

    spc_df = spc_df.rename({Species.name: SpeciesStereo.orig_name})
    spc_df = df_.map_(
        spc_df, (SpeciesStereo.orig_name, Species.amchi), Species.name, _stereo_name
    )

    # Update the SMILES strings
    def _stereo_smiles(chi):
        """Determine the stereo smiles from the AMChI."""
        return automol.amchi.smiles(chi)

    spc_df = spc_df.rename({Species.smiles: SpeciesStereo.orig_smiles})
    spc_df = df_.map_(spc_df, Species.amchi, Species.smiles, _stereo_smiles)
    return spc_df


def expand_parent_stereo(par_mech: Mechanism, exp_sub_mech: Mechanism) -> Mechanism:
    """Apply the stereoexpansion of a submechanism to a parent mechanism.

    Produces an equivalent of the parent mechanism, containing the distinct
    stereoisomers of the submechanism. The expansion is completely naive, with no
    consideration of stereospecificity, and is simply designed to allow merging of a
    stereo-expanded submechanism into a parent mechanism.

    :param par_mech: A parent mechanism
    :param exp_sub_mech: A stereo-expanded sub-mechanism
    :return: An equivalent parent mechanism, with distinct stereoisomers from the
        sub-mechanism
    """
    # 1. Species table
    #   a. Add stereo columns to par_mech species table
    col_dct = {
        Species.name: SpeciesStereo.orig_name,
        Species.smiles: SpeciesStereo.orig_smiles,
        Species.amchi: SpeciesStereo.orig_amchi,
    }
    par_spc_df = species(par_mech)
    par_spc_df = par_spc_df.rename(col_dct)

    #   b. Group by original names and isolate expanded stereoisomers
    sub_spc_df = species(exp_sub_mech)
    sub_spc_df = schema.species_table(sub_spc_df, models=(SpeciesStereo,))
    sub_spc_df = sub_spc_df.select(*col_dct.keys(), *col_dct.values())
    sub_spc_df = sub_spc_df.group_by(SpeciesRenamed.orig_name).agg(polars.all())
    sub_spc_df = sub_spc_df.filter(polars.col(Species.name).list.len() > 1)

    #   c. Form species expansion dictionary, to be used for reaction expansion
    exp_dct: dict[str, list[str]] = df_.lookup_dict(
        sub_spc_df, SpeciesRenamed.orig_name, Species.name
    )

    #   d. Join on original names, explode, and fill in non-stereoisomer columns
    exp_spc_df = par_spc_df.join(sub_spc_df, how="left", on=SpeciesStereo.orig_name)
    exp_spc_df = exp_spc_df.drop(polars.selectors.ends_with("_right"))
    exp_spc_df = exp_spc_df.explode(*col_dct.keys())
    exp_spc_df = exp_spc_df.with_columns(
        *(polars.col(k).fill_null(polars.col(v)) for k, v in col_dct.items())
    )

    # 2. Reaction table
    #   a. Identify the subset of reactions to be expanded
    par_rxn_df = reactions(par_mech)
    has_rate = ReactionRate.rate in par_rxn_df
    if not has_rate:
        rate = dict(data.rate.SimpleRate())
        par_rxn_df = par_rxn_df.with_columns(polars.lit(rate).alias(ReactionRate.rate))

    par_rxn_df = par_rxn_df.with_columns(
        polars.col(Reaction.reactants).alias(ReactionStereo.orig_reactants),
        polars.col(Reaction.products).alias(ReactionStereo.orig_products),
        polars.col(ReactionRate.rate).alias(ReactionMisc.orig_rate),
    )
    needs_exp = (
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(list(exp_dct.keys())))
        .list.any()
    )
    exp_rxn_df = par_rxn_df.filter(needs_exp)
    rem_rxn_df = par_rxn_df.filter(~needs_exp)

    #   b. Expand the reactions
    def _expand(rct0s, prd0s, rate0):
        rxn0 = data.reac.from_data(rct0s, prd0s, rate_=rate0)
        rxns = data.reac.expand_lumped_species(rxn0, exp_dct=exp_dct)
        rcts_lst = list(map(data.reac.reactants, rxns))
        prds_lst = list(map(data.reac.products, rxns))
        rates = list(map(dict, map(data.reac.rate, rxns)))
        return rcts_lst, prds_lst, rates

    cols = [Reaction.reactants, Reaction.products, ReactionRate.rate]
    dtypes = list(map(polars.List, map(exp_rxn_df.schema.get, cols)))
    exp_rxn_df = df_.map_(exp_rxn_df, cols, cols, _expand, dtype_=dtypes)
    exp_rxn_df: polars.DataFrame = exp_rxn_df.explode(cols)
    exp_rxn_df = polars.concat([rem_rxn_df, exp_rxn_df])

    if not has_rate:
        exp_rxn_df = exp_rxn_df.drop(ReactionRate.rate, ReactionMisc.orig_rate)

    return from_data(
        rxn_inp=exp_rxn_df,
        spc_inp=exp_spc_df,
        thermo_temps=thermo_temperatures(par_mech),
        rate_units=rate_units(par_mech),
    )


def drop_parent_reactions(par_mech: Mechanism, exp_sub_mech: Mechanism) -> Mechanism:
    """Drop equivalent reactions from a submechanism in a parent mechanism.

    :param par_mech: A parent mechanism
    :param exp_sub_mech: A stereo-expanded sub-mechanism
    :return: The parent mechanism, with updated rates
    """
    par_rxn_df = reactions(par_mech)
    sub_rxn_df = reactions(without_unused_species(exp_sub_mech))

    # Form species mappings onto AMChIs without stereo
    par_spc_df = species(par_mech)
    sub_spc_df = species(exp_sub_mech)
    par_spc_df: polars.DataFrame = df_.map_(
        par_spc_df, Species.amchi, "amchi0", automol.amchi.without_stereo
    )
    sub_spc_df: polars.DataFrame = df_.map_(
        sub_spc_df, Species.amchi, "amchi0", automol.amchi.without_stereo
    )
    par_key_dct = df_.lookup_dict(par_spc_df, Species.name, "amchi0")
    sub_key_dct = df_.lookup_dict(sub_spc_df, Species.name, "amchi0")
    par_spc_df = par_spc_df.drop("amchi0")
    sub_spc_df = sub_spc_df.drop("amchi0")

    # Add unique reaction keys for identifying the correspondence
    par_rxn_df = reac_table.with_reaction_key(
        par_rxn_df, "key", spc_key_dct=par_key_dct
    )
    sub_rxn_df = reac_table.with_reaction_key(
        sub_rxn_df, "key", spc_key_dct=sub_key_dct
    )

    # Remove overlapping reactions from parent mechanism
    is_in_sub = polars.col("key").is_in(sub_rxn_df["key"])
    par_rxn_df = par_rxn_df.filter(~is_in_sub)
    par_rxn_df = par_rxn_df.drop("key")

    par_mech = set_reactions(par_mech, par_rxn_df)
    return par_mech


def update_parent_species_data(
    par_mech: Mechanism, exp_sub_mech: Mechanism
) -> Mechanism:
    """Update the species data in a parent mechanism from a submechanism.

    Note: A pseudo-stereoexpansion will be applied to any to the parent mechanism for
    any species it shares with the sub-mechanism.

    :param par_mech: A parent mechanism
    :param exp_sub_mech: A stereo-expanded sub-mechanism
    :return: The parent mechanism, with updated thermochemistry
    """
    exp_par_mech = expand_parent_stereo(par_mech, exp_sub_mech)

    par_spc_df = species(exp_par_mech)
    sub_spc_df = species(exp_sub_mech)

    key = SpeciesThermo.thermo_string
    sub_therm_df = sub_spc_df.filter(polars.col(key).is_not_null())

    key_ = f"{key}_right"
    par_spc_df = par_spc_df.join(
        sub_therm_df, how="left", on=Species.name
    ).with_columns(
        polars.when(polars.col(key_).is_not_null())
        .then(polars.col(key_))
        .otherwise(polars.col(key))
        .alias(key)
    )
    par_spc_df = par_spc_df.drop(polars.selectors.ends_with("_right"))
    return set_species(exp_par_mech, par_spc_df)


def update_parent_reaction_data(
    par_mech: Mechanism, exp_sub_mech: Mechanism
) -> Mechanism:
    """Update the reaction data in a parent mechanism from a submechanism.

    Note: A pseudo-stereoexpansion will be applied to any to the parent mechanism for
    any species it shares with the sub-mechanism.

    :param par_mech: A parent mechanism
    :param exp_sub_mech: A stereo-expanded sub-mechanism
    :return: The parent mechanism, with updated thermochemistry
    """
    exp_par_mech = expand_parent_stereo(par_mech, exp_sub_mech)
    rem_par_mech = drop_parent_reactions(exp_par_mech, exp_sub_mech)
    rem_rxn_df = reactions(rem_par_mech)
    sub_rxn_df = reactions(exp_sub_mech)
    par_rxn_df = polars.concat([rem_rxn_df, sub_rxn_df], how="diagonal_relaxed")
    return set_reactions(rem_par_mech, par_rxn_df)


# comparison
def are_equivalent(mech1: Mechanism, mech2: Mechanism) -> bool:
    """Determine whether two mechanisms are equal.

    (Currently too strict -- need to figure out how to handle nested float comparisons
    in Struct columns.)

    Waiting on:
     - https://github.com/pola-rs/polars/issues/11067 (to be used with .unnest())
    and/or:
     - https://github.com/pola-rs/polars/issues/18936

    :param mech1: The first mechanism
    :param mech2: The second mechanism
    :return: `True` if they are, `False` if they aren't
    """
    same_reactions = reactions(mech1).equals(reactions(mech2))
    same_species = species(mech1).equals(species(mech2))
    return same_reactions and same_species


# read/write
def string(mech: Mechanism) -> str:
    """Write a mechanism to a JSON string.

    :param mech: A mechanism
    :return: The Mechanism JSON string
    """
    return json.dumps(dict(mech))


def from_string(mech_str: str) -> Mechanism:
    """Read a mechanism from a JSON string.

    :param mech_str: A Mechanism JSON string
    :return: The mechanism
    """
    mech_dct = json.loads(mech_str)
    return Mechanism(**mech_dct)


# display
def display(
    mech: Mechanism,
    stereo: bool = True,
    node_exclude_formulas: Sequence[str] = DEFAULT_EXCLUDE_FORMULAS,
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display the mechanism as a reaction network.

    :param mech: The mechanism
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param node_exclude_formulas: Formulas for species to be excluded as nodes
    :param out_name: The name of the HTML file for the network visualization
    :param out_dir: The name of the directory for saving the network visualization
    :param open_browser: Whether to open the browser automatically
    """
    mech_net = species_network(mech, node_exclude_formulas=node_exclude_formulas)
    net_.display(
        mech_net,
        stereo=stereo,
        out_name=out_name,
        out_dir=out_dir,
        open_browser=open_browser,
    )


def display_species(
    mech: Mechanism,
    sel_vals: Sequence[str] | None = None,
    sel_key: str = Species.name,
    stereo: bool = True,
    keys: tuple[str, ...] = (
        Species.name,
        Species.smiles,
    ),
):
    """Display the species in a mechanism.

    :param mech: The mechanism
    :param sel_vals: Select species by column value, defaults to None
    :param sel_key: The column to use for selection, defaults to Species.smiles
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    """
    # Read in the mechanism data
    spc_df: polars.DataFrame = species(mech)

    if sel_vals is not None:
        if sel_key == Species.smiles:
            sel_vals = list(map(automol.smiles.amchi, sel_vals))
            sel_key = Species.amchi
        spc_df = spc_df.filter(polars.col(sel_key).is_in(sel_vals))
        keys = (sel_key, *keys) if sel_key not in keys else keys

    def _display_species(chi, *vals):
        """Display a species."""
        # Print the requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        automol.amchi.display(chi, stereo=stereo)

    # Display the requested reactions
    spc_df = df_.map_(spc_df, (Species.amchi, *keys), None, _display_species)


def display_reactions(
    mech: Mechanism,
    eqs: Collection | None = None,
    stereo: bool = True,
    keys: Sequence[str] = (),
    spc_keys: Sequence[str] = (Species.smiles,),
):
    """Display the reactions in a mechanism.

    :param mech: The mechanism
    :param eqs: Optionally, specify specific equations to visualize
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    :param spc_keys: Optionally, translate the reactant and product names into these
        species dataframe values
    """
    # Read in the mechanism data
    spc_df: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    chi_dct = df_.lookup_dict(spc_df, Species.name, Species.amchi)
    trans_dcts = {k: df_.lookup_dict(spc_df, Species.name, k) for k in spc_keys}

    rxn_df = df_.map_(
        rxn_df,
        (Reaction.reactants, Reaction.products),
        "eq",
        data.reac.write_chemkin_equation,
    )

    if eqs is not None:
        eqs = list(map(data.reac.standardize_chemkin_equation, eqs))
        rxn_df = rxn_df.filter(polars.col("eq").is_in(eqs))

    def _display_reaction(eq, *vals):
        """Add a node to the network."""
        # Print the requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        # Display the reaction
        rchis, pchis, *_ = data.reac.read_chemkin_equation(eq, trans_dct=chi_dct)

        for key, trans_dct in trans_dcts.items():
            rvals, pvals, *_ = data.reac.read_chemkin_equation(eq, trans_dct=trans_dct)
            print(f"Species `name`=>`{key}` translation")
            print(f"  reactants = {rvals}")
            print(f"  products = {pvals}")

        if not all(isinstance(n, str) for n in rchis + pchis):
            print(f"Some ChIs missing from species table: {rchis} = {pchis}")
        else:
            automol.amchi.display_reaction(rchis, pchis, stereo=stereo)

    # Display the requested reactions
    rxn_df = df_.map_(rxn_df, ("eq", *keys), None, _display_reaction)
