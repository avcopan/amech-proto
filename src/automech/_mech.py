"""Definition and core functionality of the mechanism data structure."""

import dataclasses
import itertools
import textwrap
from collections.abc import Collection, Sequence
from pathlib import Path

import automol
import more_itertools as mit
import polars
import pyvis

# from IPython import display as ipd
from . import data, schema
from .schema import (
    Model,
    Reaction,
    ReactionRenamed,
    ReactionStereo,
    Species,
    SpeciesRenamed,
    SpeciesStereo,
)
from .util import df_


@dataclasses.dataclass
class Mechanism:
    """A chemical kinetic mechanism."""

    reactions: polars.DataFrame
    species: polars.DataFrame

    def __repr__(self):
        rxn_df_rep = textwrap.indent(repr(self.reactions), "  ")
        spc_df_rep = textwrap.indent(repr(self.species), "  ")
        rxn_rep = textwrap.indent(f"reactions=DataFrame(\n{rxn_df_rep}\n)", "  ")
        spc_rep = textwrap.indent(f"species=DataFrame(\n{spc_df_rep}\n)", "  ")
        return f"Mechanism(\n{rxn_rep},\n{spc_rep},\n)"


# constructors
def from_data(
    rxn_inp: str | polars.DataFrame,
    spc_inp: str | polars.DataFrame,
    rxn_models: Sequence[Model] = (),
    spc_models: Sequence[Model] = (),
) -> Mechanism:
    """Contruct a mechanism object from data.

    :param rxn_inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param rxn_models: Extra reaction models to validate against
    :param spc_models: Extra species models to validate against
    :param drop_spc: Drop unused species from the mechanism?
    :return: The mechanism object
    """
    rxn_df = rxn_inp if isinstance(rxn_inp, polars.DataFrame) else df_.from_csv(rxn_inp)
    spc_df = spc_inp if isinstance(spc_inp, polars.DataFrame) else df_.from_csv(spc_inp)
    rxn_df = schema.reaction_table(rxn_df, models=rxn_models)
    spc_df = schema.species_table(spc_df, models=spc_models)
    return Mechanism(reactions=rxn_df, species=spc_df)


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
    dt = schema.types([Species], data_dct.keys())
    spc_df = polars.DataFrame(data=data_dct, schema=dt)
    spc_df = schema.species_table(
        spc_df, name_dct=name_dct, spin_dct=spin_dct, charge_dct=charge_dct
    )

    # Build the reactions dataframe
    trans_dct = df_.lookup_dict(spc_df, Species.smiles, Species.name)
    rxn_smis_lst = list(map(automol.smiles.reaction_reactants_and_products, rxn_smis))
    eqs = [
        data.reac.write_chemkin_equation(rs, ps, trans_dct=trans_dct)
        for rs, ps in rxn_smis_lst
    ]
    data_dct = {Reaction.eq: eqs}
    dt = schema.types([Reaction], data_dct.keys())
    rxn_df = polars.DataFrame(data=data_dct, schema=dt)
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


# getters
def set_species(mech: Mechanism, spc_df: polars.DataFrame) -> Mechanism:
    """Set the species dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism with updated species
    """
    return from_data(reactions(mech), spc_df)


def set_reactions(mech: Mechanism, rxn_df: polars.DataFrame) -> Mechanism:
    """Set the reactions dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism with updated reactions
    """
    return from_data(rxn_df, species(mech))


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


def reacting_species_names(mech: Mechanism) -> list[str]:
    """Get the names of the species that are involved in reactions.

    :param mech: The mechanism
    :return: The names of reacting species
    """
    rxn_df = reactions(mech)
    eqs = rxn_df[Reaction.eq].to_list()
    rxn_names = [r + p for r, p, *_ in map(data.reac.read_chemkin_equation, eqs)]
    names = list(mit.unique_everseen(itertools.chain(*rxn_names)))
    return names


def expansion_name_dict(mech: Mechanism) -> dict[str, list[str]]:
    """Generate a dictionary describing the mapping of names from an expansion.

    :param mech: A mechanism whose species table has an "orig_name" column
    :return: A dictionary mapping original species names onto lists of expansion names
    """
    spc_df = schema.species_table(species(mech), models=[SpeciesRenamed])
    exp_df = spc_df.group_by(SpeciesRenamed.orig_name).agg(polars.all())
    exp_dct: dict[str, list[str]] = df_.lookup_dict(
        exp_df, SpeciesRenamed.orig_name, Species.name
    )
    exp_dct = {k: v for k, v in exp_dct.items() if len(v) > 1}
    return exp_dct


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

    def _new_name(orig_name: str) -> str:
        """Rename a species.

        :param orig_name: The original name
        :return: The new name
        """
        return name_dct.get(orig_name) if orig_name in name_dct else orig_name

    def _new_eq(orig_eq: str) -> str:
        rname0s, pname0s, coll, arrow = data.reac.read_chemkin_equation(orig_eq)
        rnames = list(map(_new_name, rname0s))
        pnames = list(map(_new_name, pname0s))
        return data.reac.write_chemkin_equation(rnames, pnames, coll, arrow)

    spc_df = species(mech)
    spc_df = spc_df.rename({Species.name: SpeciesRenamed.orig_name})
    spc_df = df_.map_(spc_df, SpeciesRenamed.orig_name, Species.name, _new_name)

    rxn_df = reactions(mech)
    rxn_df = rxn_df.rename({Reaction.eq: ReactionRenamed.orig_eq})
    rxn_df = df_.map_(rxn_df, ReactionRenamed.orig_eq, Reaction.eq, _new_eq)
    return from_data(rxn_inp=rxn_df, spc_inp=spc_df)


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
    # Read in the mechanism data
    spc_df: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    spc_names = set(spc_names)

    def _include(eq: str) -> bool:
        rct_names, prd_names, *_ = data.reac.read_chemkin_equation(eq)
        rgt_names = set(rct_names + prd_names)
        is_incl = rgt_names <= spc_names if strict else bool(rgt_names & spc_names)
        return without ^ is_incl

    rxn_df = df_.map_(rxn_df, Reaction.eq, "incl", _include)
    rxn_df = rxn_df.filter("incl").drop("incl")

    return without_unused_species(from_data(rxn_inp=rxn_df, spc_inp=spc_df))


def without_unused_species(mech: Mechanism) -> Mechanism:
    """Remove unused species from a mechanism.

    :param mech: The mechanism
    :return: The mechanism, without unused species
    """
    spc_df = species(mech)
    used_names = reacting_species_names(mech)
    spc_df = spc_df.filter(polars.col(Species.name).is_in(used_names))
    return set_species(mech, spc_df)


def expand_stereo(
    mech: Mechanism,
    enant: bool = True,
    strained: bool = False,
    drop_unused: bool = True,
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

    # Do the reaction expansion
    chi_dct: dict = df_.lookup_dict(
        spc_df, SpeciesStereo.orig_name, SpeciesStereo.orig_amchi
    )
    name_dct: dict = df_.lookup_dict(
        spc_df, (SpeciesStereo.orig_name, Species.amchi), Species.name
    )

    def _expand_amchi(orig_eq):
        """Classify a reaction and return the reaction objects."""
        rname0s, pname0s, coll, arrow = data.reac.read_chemkin_equation(orig_eq)
        rchi0s = list(map(chi_dct.get, rname0s))
        pchi0s = list(map(chi_dct.get, pname0s))
        objs = automol.reac.from_amchis(rchi0s, pchi0s, stereo=False)
        vals = []
        for obj in objs:
            sobjs = automol.reac.expand_stereo(obj, enant=enant, strained=strained)
            for sobj in sobjs:
                # Determine the AMChI
                chi = automol.reac.ts_amchi(sobj)
                # Determine the updated equation
                rchis, pchis = automol.reac.amchis(sobj)
                rnames = tuple(map(name_dct.get, zip(rname0s, rchis, strict=True)))
                pnames = tuple(map(name_dct.get, zip(pname0s, pchis, strict=True)))
                if not all(isinstance(n, str) for n in rnames + pnames):
                    return polars.Null

                eq = data.reac.write_chemkin_equation(rnames, pnames, coll, arrow)
                vals.append([eq, chi])
        return vals if vals else polars.Null

    tmp_col = "tmp"
    rxn_df = df_.map_(rxn_df, Reaction.eq, tmp_col, _expand_amchi)

    # Separate out the error cases
    err_df = rxn_df.filter(polars.col(tmp_col).is_null())
    rxn_df = rxn_df.filter(polars.col(tmp_col).is_not_null())

    # Expand the table by stereoisomers
    err_df = err_df.drop(tmp_col)
    rxn_df = rxn_df.explode(polars.col(tmp_col))

    # Split the AMChI and equation columns
    rxn_df = rxn_df.rename({Reaction.eq: ReactionStereo.orig_eq})
    rxn_df = rxn_df.with_columns(
        polars.col(tmp_col)
        .list.to_struct()
        .struct.rename_fields([Reaction.eq, ReactionStereo.amchi])
    ).unnest(tmp_col)

    mech = from_data(rxn_df, spc_df)
    err_mech = from_data(err_df, spc_df0)

    if drop_unused:
        mech = without_unused_species(mech)
        err_mech = without_unused_species(err_mech)

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


# display
def display(
    mech: Mechanism,
    stereo: bool = True,
    exclude_formulas: tuple[str, ...] = ("H*", "OH*", "O2H*", "CH*"),
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display the mechanism as a reaction network.

    :param mech: The mechanism
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param exclude: Formula strings of molecules to exclude from the  network nodes,
        using * for wildcard stoichiometry, defaults to ("H*", "OH*", "O2H*", "CH*")
    :param out_name: The name of the HTML file for the network visualization
    :param out_dir: The name of the directory for saving the network visualization
    :param open_browser: Whether to open the browser automatically
    """
    out_dir: Path = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    img_dir = Path("img")
    (out_dir / img_dir).mkdir(exist_ok=True)
    excl_fmls = tuple(map(automol.form.from_string, exclude_formulas))
    net = pyvis.network.Network(directed=True, notebook=True, cdn_resources="in_line")

    # Read in the mechanism data
    spc_df: polars.DataFrame = species(mech)
    rxn_df: polars.DataFrame = reactions(mech)

    if rxn_df.is_empty():
        print(f"The reaction network is empty. Skipping visualization...\n{mech}")
        return

    # Define some functions
    def _is_excluded(chi):
        """Determine whether a species is excluded."""
        fml = automol.amchi.formula(chi)
        return any(automol.form.match(fml, e) for e in excl_fmls)

    spc_df = df_.map_(spc_df, Species.amchi, "excluded", _is_excluded)
    excl_names = list(spc_df.filter(polars.col("excluded"))[Species.name])

    def _image_path(chi):
        """Create an SVG molecule drawing and return the path."""
        gra = automol.amchi.graph(chi, stereo=stereo)
        chk = automol.amchi.amchi_key(chi)
        svg_str = automol.graph.svg_string(gra, image_size=100)

        path = img_dir / f"{chk}.svg"
        with open(out_dir / path, mode="w") as file:
            file.write(svg_str)

        return str(path)

    def _add_node(name, smi, path):
        """Add a node to the network."""
        if name not in excl_names:
            net.add_node(name, shape="image", image=path, title=smi)

    def _add_edge(eq):
        """Add an edge to the network."""
        rnames, pnames, *_ = data.reac.read_chemkin_equation(eq)
        for rname, pname in itertools.product(rnames, pnames):
            if rname not in excl_names and pname not in excl_names:
                net.add_edge(rname, pname, title=eq)

    # Generate SVG drawings with paths
    spc_df = df_.map_(spc_df, Species.amchi, "image_path", _image_path)
    # Add nodes to the network
    spc_df = df_.map_(
        spc_df, (Species.name, Species.smiles, "image_path"), None, _add_node
    )
    # Add edges to the network
    rxn_df = df_.map_(rxn_df, Reaction.eq, None, _add_edge)
    # Visualize the network
    net.write_html(str(out_dir / out_name), open_browser=open_browser)


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
    keys: tuple[str, ...] = (Reaction.eq,),
    spc_keys: tuple[str, ...] = (Species.smiles,),
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

    if eqs is not None:
        eqs = list(map(data.reac.standardize_chemkin_equation, eqs))
        rxn_df = rxn_df.filter(polars.col(Reaction.eq).is_in(eqs))

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
    rxn_df = df_.map_(rxn_df, (Reaction.eq, *keys), None, _display_reaction)
