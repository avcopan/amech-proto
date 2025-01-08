"""Definition and core functionality of mechanism data structure."""

import dataclasses
import itertools
import json
import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence

import automol
import more_itertools as mit
import polars

from . import data, reac_table, schema, spec_table
from . import net as net_
from .schema import (
    Model,
    Reaction,
    ReactionMisc,
    ReactionRate,
    ReactionRenamed,
    ReactionSorted,
    ReactionStereo,
    Species,
    SpeciesRenamed,
    SpeciesStereo,
    SpeciesThermo,
)
from .util import df_


@dataclasses.dataclass
class Mechanism:
    """Chemical kinetic mechanism."""

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
        """Iterate over mechanism's dictionary representation."""
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
    """Construct mechanism object from data.

    :param rxn_inp: Reactions table, as CSV file path or DataFrame
    :param spc_inp: Species table, as CSV file path or DataFrame
    :param rxn_models: Extra reaction models to validate against
    :param spc_models: Extra species models to validate against
    :param fail_on_error: Whether to raise exception if there is inconsistency
    :return: Mechanism object
    """
    spc_df = spc_inp if isinstance(spc_inp, polars.DataFrame) else df_.from_csv(spc_inp)
    rxn_df = rxn_inp if isinstance(rxn_inp, polars.DataFrame) else df_.from_csv(rxn_inp)
    spc_df = schema.species_table(spc_df, model_=spc_models)
    rxn_df, _ = schema.reaction_table(
        rxn_df, model_=rxn_models, spc_df=spc_df, fail_on_error=fail_on_error
    )
    mech = Mechanism(
        reactions=rxn_df,
        species=spc_df,
        thermo_temps=thermo_temps,
        rate_units=rate_units,
    )
    return mech


def from_network(net: net_.Network) -> Mechanism:
    """Generate mechanism from reaction network.

    :param net: Reaction network
    :return: Mechanism
    """
    spc_data = list(
        itertools.chain(*(d[net_.Key.species] for *_, d in net.nodes.data()))
    )
    rxn_data = [d for *_, d in net.edges.data()]

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
    spc_df = spc_df.drop(net_.Key.id, strict=False)
    rxn_df = rxn_df.drop(net_.Key.id, strict=False)
    return from_data(rxn_inp=rxn_df, spc_inp=spc_df, fail_on_error=False)


def from_smiles(
    spc_smis: Sequence[str] = (),
    rxn_smis: Sequence[str] = (),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
) -> Mechanism:
    """Generate mechanism using SMILES strings for species names.

    If `name_dct` is `None`, CHEMKIN names will be auto-generated.

    :param spc_smis: Species SMILES strings
    :param rxn_smis: Optionally, reaction SMILES strings
    :param name_dct: Optionally, specify name for some molecules
    :param spin_dct: Optionally, specify spin state (2S) for some molecules
    :param charge_dct: Optionally, specify charge for some molecules
    :return: Mechanism
    """
    name_dct = {} if name_dct is None else name_dct
    spin_dct = {} if spin_dct is None else spin_dct
    charge_dct = {} if charge_dct is None else charge_dct

    # Add in any missing species from reaction SMILES
    spc_smis_by_rxn = [
        rs + ps
        for (rs, ps) in map(automol.smiles.reaction_reactants_and_products, rxn_smis)
    ]
    spc_smis = list(mit.unique_everseen(itertools.chain(spc_smis, *spc_smis_by_rxn)))

    # Build species dataframe
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

    # Build reactions dataframe
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
    """Get species DataFrame for mechanism.

    :param mech: Mechanism
    :return: Mechanism's species DataFrame
    """
    return mech.species


species_ = species


def reactions(mech: Mechanism) -> polars.DataFrame:
    """Get reactions DataFrame for mechanism.

    :param mech: Mechanism
    :return: Mechanism's reactions DataFrame
    """
    return mech.reactions


reactions_ = reactions


def thermo_temperatures(mech: Mechanism) -> tuple[float, float, float] | None:
    """Get thermo temperatures for mechanism.

    :param mech: Mechanism
    :return: Thermo temperatures
    """
    return mech.thermo_temps


thermo_temperatures_ = thermo_temperatures


def rate_units(mech: Mechanism) -> tuple[str, str] | None:
    """Get rate units for mechanism.

    :param mech: Mechanism
    :return: Rate units
    """
    return mech.rate_units


rate_units_ = rate_units


# setters
def set_species(mech: Mechanism, spc_df: polars.DataFrame) -> Mechanism:
    """Set species DataFrame for mechanism.

    :param mech: Mechanism
    :param spc_df: New species DataFrame
    :return: Mechanism with updated species
    """
    return from_data(
        rxn_inp=reactions(mech),
        spc_inp=spc_df,
        thermo_temps=thermo_temperatures(mech),
        rate_units=rate_units(mech),
    )


def set_reactions(mech: Mechanism, rxn_df: polars.DataFrame) -> Mechanism:
    """Set reactions DataFrame for mechanism.

    :param mech: Mechanism
    :param rxn_df: New reactions DataFrame
    :return: Mechanism with updated reactions
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
    """Set thermo temperatures for mechanism.

    :param mech: Mechanism
    :param temps: New thermo temperatures
    :return: Mechanism with updated thermo temperatures
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
    """Set rate units for mechanism.

    :param mech: Mechanism
    :param units: New rate units
    :param scale_rates: Scale rates if changing units?
    :return: Mechanism with updated rate units
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


def update(
    mech: Mechanism,
    rxn_df: polars.DataFrame | None = None,
    spc_df: polars.DataFrame | None = None,
    thermo_temps: tuple[float, float, float] | None = None,
    rate_units: tuple[str, str] | None = None,
) -> Mechanism:
    """Update mechanism data.

    :param rxn_df: Reactions DataFrame
    :param spc_df: Species DataFrame
    :param thermo_temps: Thermodynamic temperatures
    :param rate_units: Rate units
    :return: Mechanism
    """
    return from_data(
        rxn_inp=reactions(mech) if rxn_df is None else rxn_df,
        spc_inp=species(mech) if spc_df is None else spc_df,
        thermo_temps=(
            thermo_temperatures(mech) if thermo_temps is None else thermo_temps
        ),
        rate_units=rate_units_(mech) if rate_units is None else rate_units,
    )


# properties
def species_count(mech: Mechanism) -> int:
    """Get number of species in mechanism.

    :param mech: Mechanism
    :return: Number of species
    """
    return df_.count(species(mech))


def reaction_count(mech: Mechanism) -> int:
    """Get number of reactions in mechanism.

    :param mech: Mechanism
    :return: Number of reactions
    """
    return df_.count(reactions(mech))


def reagents(mech: Mechanism) -> list[list[str]]:
    """Get sets of reagents in mechanism.

    :param mech: Mechanism
    :return: Sets of reagents
    """
    return reac_table.reagents(reactions(mech))


def species_names(
    mech: Mechanism,
    rxn_only: bool = False,
    formulas: Sequence[str] | None = None,
    exclude_formulas: Sequence[str] = (),
) -> list[str]:
    """Get names of species in mechanism.

    :param mech: Mechanism
    :param rxn_only: Only include species that are involved in reactions?
    :param formulas: Formula strings of species to include, using * for wildcard
        stoichiometry
    :param exclude_formulas: Formula strings of species to exclude, using * for wildcard
        stoichiometry
    :return: Species names
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
    """Get reactants of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction reactants
    """
    rxn_df = reactions(mech)
    return rxn_df[Reaction.reactants].to_list()


def reaction_products(mech: Mechanism) -> list[list[str]]:
    """Get products of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction products
    """
    rxn_df = reactions(mech)
    return rxn_df[Reaction.products].to_list()


def reaction_reactants_and_products(
    mech: Mechanism,
) -> list[tuple[list[str], list[str]]]:
    """Get reactants and products of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction reactants and products
    """
    rxn_df = reactions(mech)
    return rxn_df[[Reaction.reactants, Reaction.products]].rows()


def reaction_equations(mech: Mechanism) -> list[str]:
    """Get equations of reactions in mechanism.

    :param mech: Mechanism
    :return: Reaction equations
    """
    rps = reaction_reactants_and_products(mech)
    return list(itertools.starmap(data.reac.write_chemkin_equation, rps))


def reaction_species_names(mech: Mechanism) -> list[str]:
    """Get names of all species that participate in reactions.

    :param mech: Mechanism
    :return: Reaction species names
    """
    eqs = reaction_equations(mech)
    rxn_names = [r + p for r, p, *_ in map(data.reac.read_chemkin_equation, eqs)]
    return list(mit.unique_everseen(itertools.chain(*rxn_names)))


def rename_dict(mech1: Mechanism, mech2: Mechanism) -> tuple[dict[str, str], list[str]]:
    """Generate dictionary for renaming species names from one mechanism to another.

    :param mech1: Mechanism with original names
    :param mech2: Mechanism with desired names
    :return: Dictionary mapping names from `mech1` to those in `mech2`, and list
        of names from `mech1` that are missing in `mech2`
    """
    match_cols = [Species.amchi, Species.spin, Species.charge]

    # Read in species and names
    spc1_df = species(mech1)
    spc1_df = spc1_df.rename({Species.name: SpeciesRenamed.orig_name})

    spc2_df = species(mech2)
    spc2_df = spc2_df.select([Species.name, *match_cols])

    # Get names from first mechanism that are included/excluded in second
    incl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="inner")
    excl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="anti")

    name_dct = df_.lookup_dict(incl_spc_df, SpeciesRenamed.orig_name, Species.name)
    missing_names = excl_spc_df[SpeciesRenamed.orig_name].to_list()
    return name_dct, missing_names


def network(mech: Mechanism) -> net_.Network:
    """Generate reaction network representation of mechanism.

    :param mech: Mechanism
    :return: Reaction network
    """
    spc_df = species(mech)
    rxn_df = reactions(mech)

    # Double-check that reagents are sorted
    rxn_df = schema.reaction_table_with_sorted_reagents(rxn_df)

    # Add species and reaction indices
    spc_df = df_.with_index(spc_df, net_.Key.id)
    rxn_df = df_.with_index(rxn_df, net_.Key.id)

    # Exluded species
    rgt_names = list(
        itertools.chain(
            *rxn_df[Reaction.reactants].to_list(), *rxn_df[Reaction.products].to_list()
        )
    )
    excl_spc_df = spc_df.filter(~polars.col(Species.name).is_in(rgt_names))

    # Get dataframe of reagents
    rgt_col = "reagents"
    rgt_exprs = [
        rxn_df.select(polars.col(Reaction.reactants).alias(rgt_col), Reaction.formula),
        rxn_df.select(polars.col(Reaction.products).alias(rgt_col), Reaction.formula),
        excl_spc_df.select(
            polars.concat_list(Species.name).alias(rgt_col), Species.formula
        ),
    ]
    rgt_df = polars.concat(rgt_exprs, how="vertical_relaxed").group_by(rgt_col).first()

    # Append species data to reagents dataframe
    names = spc_df[Species.name]
    datas = spc_df.to_struct()
    expr = polars.element().replace_strict(names, datas)
    rgt_df = rgt_df.with_columns(
        polars.col(rgt_col).list.eval(expr).alias(net_.Key.species)
    )

    # Build network object
    def _node_data_from_dict(dct: dict[str, object]):
        key = tuple(dct.get(rgt_col))
        return (key, dct)

    def _edge_data_from_dict(dct: dict[str, object]):
        key1 = tuple(dct.get(Reaction.reactants))
        key2 = tuple(dct.get(Reaction.products))
        return (key1, key2, dct)

    return net_.from_data(
        node_data=list(map(_node_data_from_dict, rgt_df.to_dicts())),
        edge_data=list(map(_edge_data_from_dict, rxn_df.to_dicts())),
    )


def apply_network_function(
    mech: Mechanism, func: Callable, *args, **kwargs
) -> Mechanism:
    """Apply network function to mechanism.

    :param mech: Mechanism
    :param func: Function
    :param *args: Function arguments
    :param **kwargs: Function keyword arguments
    :return: Mechanism
    """
    mech0 = mech

    col_idx = df_.temp_column()
    spc_df = df_.with_index(species(mech0), name=col_idx)
    rxn_df = df_.with_index(reactions(mech0), name=col_idx)
    mech0 = update(mech0, rxn_df=rxn_df, spc_df=spc_df)
    net0 = network(mech0)
    net = func(net0, *args, **kwargs)
    spc_idxs = net_.species_values(net, col_idx)
    rxn_idxs = net_.edge_values(net, col_idx)
    spc_df = spc_df.filter(polars.col(col_idx).is_in(spc_idxs)).drop(col_idx)
    rxn_df = rxn_df.filter(polars.col(col_idx).is_in(rxn_idxs)).drop(col_idx)
    return update(mech0, rxn_df=rxn_df, spc_df=spc_df)


# transformations
def rename(
    mech: Mechanism, name_dct: dict[str, str], drop_missing: bool = False
) -> Mechanism:
    """Rename species in mechanism.

    :param mech: Mechanism
    :param name_dct: Dictionary mapping current species names to new species names
    :param drop_missing: Drop missing species from mechanism? Otherwise, they are
        retained with their original names
    :return: Mechanism with updated species names
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
    """Clear reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism without reactions
    """
    return set_reactions(mech, reactions(mech).clear())


def add_reactions(mech: Mechanism, rxn_df: polars.DataFrame) -> Mechanism:
    """Add reactions from DataFrame to mechanism.

    :param mech: Mechanism
    :param rxn_df: Reactions DataFrame
    :return: Mechanism with added reactions
    """
    rxn_df0 = reactions(mech)
    return set_reactions(mech, polars.concat([rxn_df0, rxn_df], how="diagonal_relaxed"))


def select_pes(
    mech: Mechanism, formula_: str | dict | Sequence[str | dict], exclude: bool = False
) -> Mechanism:
    """Select (or exclude) PES by formula(s).

    :param mech: Mechanism
    :param formula_: PES formula(s) to include or exclude
    :param exclude: Whether to exclude or include the formula(s)
    :return: Mechanism
    """
    rxn_df = reac_table.select_pes(reactions(mech), formula_, exclude=exclude)
    return without_unused_species(set_reactions(mech, rxn_df))


def neighborhood(
    mech: Mechanism, species_names: Sequence[str], radius: int = 1
) -> Mechanism:
    """Determine neighborhood of set of species.

    :param mech: Mechanism
    :param species_names: Names of species
    :param radius: Maximum distance of neighbors to include, defaults to 1
    :return: Neighborhood mechanism
    """
    return apply_network_function(
        mech, net_.neighborhood, species_names=species_names, radius=radius
    )


def with_species(
    mech: Mechanism, spc_names: Sequence[str] = (), strict: bool = False
) -> Mechanism:
    """Extract submechanism including species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    return _with_or_without_species(
        mech=mech, spc_names=spc_names, without=False, strict=strict
    )


def without_species(mech: Mechanism, spc_names: Sequence[str] = ()) -> Mechanism:
    """Extract submechanism excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be excluded
    :return: Submechanism
    """
    return _with_or_without_species(mech=mech, spc_names=spc_names, without=True)


def _with_or_without_species(
    mech: Mechanism,
    spc_names: Sequence[str] = (),
    without: bool = False,
    strict: bool = False,
) -> Mechanism:
    """Extract submechanism containing or excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included or excluded
    :param without: Extract submechanism *without* these species?
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    # Build appropriate filtering expression
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
    """Remove unused species from mechanism.

    :param mech: Mechanism
    :return: Mechanism without unused species
    """
    spc_df = species(mech)
    used_names = species_names(mech, rxn_only=True)
    spc_df = spc_df.filter(polars.col(Species.name).is_in(used_names))
    return set_species(mech, spc_df)


def without_duplicate_reactions(mech: Mechanism) -> Mechanism:
    """Remove duplicate reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism without duplicate reactions
    """
    col_tmp = df_.temp_column()
    rxn_df = reactions(mech)
    rxn_df = reac_table.with_reaction_key(rxn_df, col_name=col_tmp)
    rxn_df = rxn_df.unique(col_tmp, maintain_order=True)
    rxn_df = rxn_df.drop(col_tmp)
    return set_reactions(mech, rxn_df)


def with_rates(mech: Mechanism) -> Mechanism:
    """Add dummy placeholder rates to this Mechanism, if missing.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Mechanism
    :return: Mechanism with dummy rates, if missing
    """
    rxn_df = reactions(mech)
    return set_reactions(mech, reac_table.with_rates(rxn_df))


def expand_stereo(
    mech: Mechanism,
    enant: bool = True,
    strained: bool = False,
    distinct_ts: bool = True,
) -> tuple[Mechanism, Mechanism]:
    """Expand stereochemistry for mechanism.

    :param mech: Mechanism
    :param enant: Distinguish between enantiomers?, defaults to True
    :param strained: Include strained stereoisomers?
    :param distinct_ts: Include duplicate reactions for distinct TSs?
    :return: Mechanism with classified reactions, and one with unclassified
    """
    # Read in mechanism data
    spc_df0: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    # Do species expansion
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
    rxn_df = rxn_df.drop(col_dct.values(), strict=False)
    rxn_df = rxn_df.rename(col_dct)

    # Define expansion function
    name_dct: dict = df_.lookup_dict(
        spc_df, (SpeciesStereo.orig_name, Species.amchi), Species.name
    )

    def _expand_reaction(rchi0s, pchi0s, rname0s, pname0s):
        """Classify reaction and return reaction objects."""
        objs = automol.reac.from_amchis(rchi0s, pchi0s, stereo=False)
        rnames_lst = []
        pnames_lst = []
        ts_amchis = []
        for obj in objs:
            sobjs = automol.reac.expand_stereo(obj, enant=enant, strained=strained)
            for sobj in sobjs:
                # Determine AMChI
                ts_amchi = automol.reac.ts_amchi(sobj)
                # Determine updated equation
                rchis, pchis = automol.reac.amchis(sobj)
                rnames = tuple(map(name_dct.get, zip(rname0s, rchis, strict=True)))
                pnames = tuple(map(name_dct.get, zip(pname0s, pchis, strict=True)))
                if not all(isinstance(n, str) for n in rnames + pnames):
                    return ([], [])

                rnames_lst.append(rnames)
                pnames_lst.append(pnames)
                ts_amchis.append(ts_amchi)
        return rnames_lst, pnames_lst, ts_amchis

    # Do expansion
    cols_in = (
        "ramchis",
        "pamchis",
        ReactionStereo.orig_reactants,
        ReactionStereo.orig_products,
    )
    cols_out = (Reaction.reactants, Reaction.products, ReactionStereo.amchi)
    rxn_df = df_.map_(rxn_df, cols_in, cols_out, _expand_reaction)

    # Separate out error cases
    err_df = rxn_df.filter(polars.col(Reaction.reactants).list.len() == 0)
    rxn_df = rxn_df.filter(polars.col(Reaction.reactants).list.len() != 0)

    # Expand table by stereoisomers
    err_df = err_df.drop(ReactionStereo.amchi, *col_dct.keys()).rename(
        dict(map(reversed, col_dct.items()))
    )
    rxn_df = rxn_df.explode(Reaction.reactants, Reaction.products, ReactionStereo.amchi)

    # Form new mechanisms
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

    if not distinct_ts:
        mech = without_duplicate_reactions(mech)

    return mech, err_mech


def _expand_species_stereo(
    spc_df: polars.DataFrame, enant: bool = True, strained: bool = False
) -> polars.DataFrame:
    """Stereoexpand species from mechanism.

    :param spc_df: Species table, as DataFrame
    :param enant: Distinguish between enantiomers?
    :param strained: Include strained stereoisomers?
    :return: Stereoexpanded species table
    """

    # Do species expansion based on AMChIs
    def _expand_amchi(chi):
        """Expand stereo for AMChIs."""
        return automol.amchi.expand_stereo(chi, enant=enant)

    spc_df = spc_df.rename({Species.amchi: SpeciesStereo.orig_amchi})
    spc_df = df_.map_(spc_df, SpeciesStereo.orig_amchi, Species.amchi, _expand_amchi)
    spc_df = spc_df.explode(polars.col(Species.amchi))

    # Update species names
    def _stereo_name(orig_name, chi):
        """Determine stereo name from AMChI."""
        return automol.amchi.chemkin_name(chi, root_name=orig_name)

    spc_df = spc_df.rename({Species.name: SpeciesStereo.orig_name})
    spc_df = df_.map_(
        spc_df, (SpeciesStereo.orig_name, Species.amchi), Species.name, _stereo_name
    )

    # Update SMILES strings
    def _stereo_smiles(chi):
        """Determine stereo smiles from AMChI."""
        return automol.amchi.smiles(chi)

    spc_df = spc_df.rename({Species.smiles: SpeciesStereo.orig_smiles})
    spc_df = df_.map_(spc_df, Species.amchi, Species.smiles, _stereo_smiles)
    return spc_df


def expand_parent_stereo(par_mech: Mechanism, exp_sub_mech: Mechanism) -> Mechanism:
    """Apply stereoexpansion of submechanism to parent mechanism.

    Produces equivalent of parent mechanism, containing distinct
    stereoisomers of submechanism. Expansion is completely naive, with no
    consideration of stereospecificity, and is simply designed to allow merging of
    stereo-expanded submechanism into parent mechanism.

    :param par_mech: Parent mechanism
    :param exp_sub_mech: Stereo-expanded sub-mechanism
    :return: Equivalent parent mechanism, with distinct stereoisomers from
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
    sub_spc_df = schema.species_table(sub_spc_df, model_=(SpeciesStereo,))
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
    #   a. Identify subset of reactions to be expanded
    par_rxn_df = reactions(par_mech)
    has_rate = ReactionRate.rate in par_rxn_df
    par_rxn_df = reac_table.with_rates(par_rxn_df)

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

    #   b. Expand reactions
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
        exp_rxn_df = reac_table.without_rates(exp_rxn_df)
        exp_rxn_df = exp_rxn_df.drop(ReactionMisc.orig_rate)

    return from_data(
        rxn_inp=exp_rxn_df,
        spc_inp=exp_spc_df,
        thermo_temps=thermo_temperatures(par_mech),
        rate_units=rate_units(par_mech),
    )


def drop_parent_reactions(par_mech: Mechanism, exp_sub_mech: Mechanism) -> Mechanism:
    """Drop equivalent reactions from submechanism in parent mechanism.

    :param par_mech: Parent mechanism
    :param exp_sub_mech: Stereo-expanded sub-mechanism
    :return: Parent mechanism with updated rates
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

    # Add unique reaction keys for identifying correspondence
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
    """Update species data in parent mechanism from submechanism.

    Note: Pseudo-stereoexpansion will be applied to parent mechanism for any
    species it shares with sub-mechanism.

    :param par_mech: Parent mechanism
    :param exp_sub_mech: Stereo-expanded sub-mechanism
    :return: Parent mechanism with updated thermochemistry
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
    """Update reaction data in parent mechanism from submechanism.

    Note: Pseudo-stereoexpansion will be applied to parent mechanism for any
    species it shares with sub-mechanism.

    :param par_mech: Parent mechanism
    :param exp_sub_mech: Stereo-expanded sub-mechanism
    :return: Parent mechanism with updated thermochemistry
    """
    exp_par_mech = expand_parent_stereo(par_mech, exp_sub_mech)
    rem_par_mech = drop_parent_reactions(exp_par_mech, exp_sub_mech)
    rem_rxn_df = reactions(rem_par_mech)
    sub_rxn_df = reactions(exp_sub_mech)
    par_rxn_df = polars.concat([rem_rxn_df, sub_rxn_df], how="diagonal_relaxed")
    return set_reactions(rem_par_mech, par_rxn_df)


# building
ReagentValue_ = str | Sequence[str] | None


def enumerate_reactions_from_smarts(
    mech: Mechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | Mapping[int, ReagentValue_] | None = None,
    spc_key_: str | Sequence[str] = Species.name,
) -> Mechanism:
    """Enumerate reactions for mechanism based on SMARTS reaction template.

    Reactants can be specified as lists or dictionaries by position in the SMARTS
    template. If unspecified, all species in the mechanism will be used.

    :param mech: Mechanism
    :param smarts: SMARTS reaction template
    :param rcts_: Reactants to be used in enumeration
    :param key_: Species column key(s) for identifying reactants and products
    :return: Mechanism with enumerated reactions
    """
    rcts_ = {} if rcts_ is None else rcts_
    rcts_ = dict(enumerate(rcts_)) if isinstance(rcts_, Sequence) else rcts_
    rcts_ = {k: [v] if isinstance(v, str) else v for k, v in rcts_.items()}

    # Determine original species list
    spc_df0 = species_(mech)
    spc_dct0 = spec_table.rows_dict(spc_df0)
    spc_names = list(spc_dct0.keys())

    # Determine reactant names and extended species list
    nrcts = automol.smarts.reactant_count(smarts)
    rcts_lst = []
    for idx in range(nrcts):
        if idx in rcts_:
            row_dct = spec_table.rows_dict(
                spc_df0, rcts_[idx], key_=spc_key_, try_fill=True
            )
            spc_dct0.update(row_dct)
            rcts_lst.append(list(row_dct.keys()))
        else:
            rcts_lst.append(spc_names)

    # Enumerate reactions
    rxn_rows = []
    spc_dct = {}
    for rcts in itertools.product(*rcts_lst):
        rct_chis = [spc_dct0.get(n).get(Species.amchi) for n in rcts]
        for rxn in automol.reac.enum.from_amchis(smarts, rct_chis):
            _, prd_chis = automol.reac.amchis(rxn)
            dct = spec_table.rows_dict(
                spc_df0, prd_chis, key_=Species.amchi, try_fill=True
            )
            spc_dct0.update(dct)
            prds = list(dct.keys())
            rxn_rows.append({Reaction.reactants: rcts, Reaction.products: prds})
            spc_dct.update({n: spc_dct0.get(n) for n in rcts})
            spc_dct.update({n: spc_dct0.get(n) for n in prds})

    # Build dataframes
    spc_df = polars.DataFrame(list(spc_dct.values()))
    rxn_df = polars.DataFrame(rxn_rows)
    return from_data(rxn_inp=rxn_df, spc_inp=spc_df)


# sorting
def with_sort_data(mech: Mechanism) -> Mechanism:
    """Add columns to sort mechanism by species and reactions.

    :param mech: Mechanism
    :return: Mechanism with sort columns
    """
    # Sort species by formula
    spc_df = spec_table.sort_by_formula(species(mech))
    mech = set_species(mech, spc_df)

    # Sort reactions by shape and by reagent names
    idx_col = df_.temp_column()
    rxn_df = reactions(mech).sort(
        polars.col(Reaction.reactants).list.len(),
        polars.col(Reaction.products).list.len(),
        polars.col(Reaction.reactants).list.to_struct(),
        polars.col(Reaction.products).list.to_struct(),
    )
    rxn_df = df_.with_index(rxn_df, idx_col)
    mech = set_reactions(mech, rxn_df)

    # Generate sort data from network
    srt_dct = net_.sort_data(network(mech), idx_col)
    srt_data = [
        {
            idx_col: i,
            ReactionSorted.pes: p,
            ReactionSorted.subpes: s,
            ReactionSorted.channel: c,
        }
        for i, (p, s, c) in srt_dct.items()
    ]
    srt_schema = {idx_col: polars.UInt32, **schema.types([ReactionSorted])}
    srt_df = polars.DataFrame(srt_data, schema=srt_schema)

    # Add sort data to reactions dataframe and sort
    rxn_df = rxn_df.join(srt_df, on=idx_col, how="left")
    rxn_df = rxn_df.drop(idx_col)
    rxn_df = rxn_df.sort(
        ReactionSorted.pes, ReactionSorted.subpes, ReactionSorted.channel
    )
    return set_reactions(mech, rxn_df)


# comparison
def are_equivalent(mech1: Mechanism, mech2: Mechanism) -> bool:
    """Determine whether two mechanisms are equivalent.

    (Currently too strict -- need to figure out how to handle nested float comparisons
    in Struct columns.)

    Waiting on:
     - https://github.com/pola-rs/polars/issues/11067 (to be used with .unnest())
    and/or:
     - https://github.com/pola-rs/polars/issues/18936

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :return: `True` if they are, `False` if they aren't
    """
    same_reactions = reactions(mech1).equals(reactions(mech2))
    same_species = species(mech1).equals(species(mech2))
    return same_reactions and same_species


# read/write
def string(mech: Mechanism) -> str:
    """Write mechanism to JSON string.

    :param mech: Mechanism
    :return: Mechanism JSON string
    """
    return json.dumps(dict(mech))


def from_string(mech_str: str) -> Mechanism:
    """Read mechanism from JSON string.

    :param mech_str: Mechanism JSON string
    :return: Mechanism
    """
    mech_dct = json.loads(mech_str)
    return Mechanism(**mech_dct)


# display
def display(
    mech: Mechanism,
    stereo: bool = True,
    color_subpes: bool = True,
    species_centered: bool = False,
    exclude_formulas: Sequence[str] = net_.DEFAULT_EXCLUDE_FORMULAS,
    height: str = "750px",
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display mechanism as reaction network.

    :param mech: Mechanism
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param color_subpes: Add distinct colors to different PESs
    :param species_centered: Display as species-centered network?
    :param exclude_formulas: If species-centered, exclude these species from display
    :param height: Control height of frame
    :param out_name: Name of HTML file for network visualization
    :param out_dir: Name of directory for saving network visualization
    :param open_browser: Whether to open browser automatically
    """
    net_.display(
        net=network(mech),
        stereo=stereo,
        color_subpes=color_subpes,
        species_centered=species_centered,
        exclude_formulas=exclude_formulas,
        height=height,
        out_name=out_name,
        out_dir=out_dir,
        open_browser=open_browser,
    )


def display_species(
    mech: Mechanism,
    spc_vals_: Sequence[str] | None = None,
    spc_key_: str | Sequence[str] = Species.name,
    stereo: bool = True,
    keys: tuple[str, ...] = (
        Species.name,
        Species.smiles,
    ),
):
    """Display species in mechanism.

    :param mech: Mechanism
    :param vals_: Species column value(s) list for selection
    :param key_: Species column key(s) for selection
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    """
    # Read in mechanism data
    spc_df: polars.DataFrame = species_(mech)

    if spc_vals_ is not None:
        spc_df = spec_table.filter(spc_df, vals_=spc_vals_, key_=spc_key_)
        id_ = [spc_key_] if isinstance(spc_key_, str) else spc_key_
        keys = [*id_, *(k for k in keys if k not in id_)]

    def _display_species(chi, *vals):
        """Display a species."""
        # Print requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        automol.amchi.display(chi, stereo=stereo)

    # Display requested reactions
    spc_df = df_.map_(spc_df, (Species.amchi, *keys), None, _display_species)


def display_reactions(
    mech: Mechanism,
    eqs: Collection | None = None,
    stereo: bool = True,
    keys: Sequence[str] = (),
    spc_keys: Sequence[str] = (Species.smiles,),
):
    """Display reactions in mechanism.

    :param mech: Mechanism
    :param eqs: Optionally, specify specific equations to visualize
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    :param spc_keys: Optionally, translate reactant and product names into these
        species dataframe values
    """
    # Read in mechanism data
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
        """Add a node to network."""
        # Print requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        # Display reaction
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

    # Display requested reactions
    rxn_df = df_.map_(rxn_df, ("eq", *keys), None, _display_reaction)
