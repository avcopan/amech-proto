"""Definition and core functionality of the mechanism data structure."""

import dataclasses
import itertools
import textwrap
from collections.abc import Sequence
from pathlib import Path

import automol
import polars
import pyvis

# from IPython import display as ipd
from . import data, schema
from .schema import Reaction, Species
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
def from_data(inp, spc_inp) -> Mechanism:
    """Contruct a mechanism object from data.

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param validate: Validate the data?
    :return: The mechanism object
    """
    rxn_df = df_.from_csv(inp) if isinstance(inp, str) else inp
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp
    rxn_df = schema.reaction_table(rxn_df)
    spc_df = schema.species_table(spc_df)
    return Mechanism(reactions=rxn_df, species=spc_df)


def from_smiles(
    smis: Sequence[str],
    rxn_smis: Sequence[str] = (),
    name_dct: dict[str, str] | None = None,
    spin_dct: dict[str, int] | None = None,
    charge_dct: dict[str, int] | None = None,
) -> Mechanism:
    """Generate a mechanism, using SMILES strings for the species names.

    If `name_dct` is `None`, CHEMKIN names will be auto-generated.

    :param smis: The species SMILES strings
    :param rxn_smis: Optionally, the reaction SMILES strings
    :param name_dct: Optionally, specify the name for some molecules
    :param spin_dct: Optionally, specify the spin state (2S) for some molecules
    :param charge_dct: Optionally, specify the charge for some molecules
    :return: The mechanism
    """
    name_dct = {} if name_dct is None else name_dct
    spin_dct = {} if spin_dct is None else spin_dct
    charge_dct = {} if charge_dct is None else charge_dct

    # Build the species dataframe
    chis = list(map(automol.smiles.amchi, smis))
    chi_dct = dict(zip(smis, chis, strict=True))
    name_dct = {chi_dct[k]: v for k, v in name_dct.items() if k in smis}
    spin_dct = {chi_dct[k]: v for k, v in spin_dct.items() if k in smis}
    charge_dct = {chi_dct[k]: v for k, v in charge_dct.items() if k in smis}
    data_dct = {Species.smiles: smis, Species.amchi: chis}
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


# transformations
# def grow(
#     mech: Mechanism,
#     rxn_type: str,
#     rct1s: Sequence[str] | None = None,
#     rct2s: Sequence[str] | None = None,
#     prds: Sequence[str] | None = None,
#     id_: str = Species.smiles,
#     unimol: bool = True,
#     bimol: bool = True,
# ) -> Mechanism:
#     """Grow a mechanism by enumerating and adding reactions.

#     :param mech: The mechanism
#     :param rxn_type: The reaction type to enumerate
#     :param rct1s: Optionally, require one reagent to match one of these
#     :param rct2s: Optionally, require a second reagent to match one of these
#     :param prds: Optionally, require a product to match one of these identifiers
#     :param id_: The identifier type for species lists, 'smi' or 'chi'
#     :param unimol: Include unimolecular reactions?
#     :param bimol: Include bimolecular reactions?
#     """
#     spc_df = mech.species
#     chi_dct = df_.lookup_dict(spc_df, id_, Species.amchi)
#     chi_conv_ = automol.smiles.amchi if id_ == Species.smiles else lambda x: x

#     r1_chis = list(spc_df[Species.amchi] if rct1s is None else map(chi_dct.get,rct1s))
#     r2_chis = list(spc_df[Species.amchi] if rct2s is None else map(chi_dct.get,rct2s))
#     p_chis = None if prds is None else list(map(chi_conv_, prds))

#     rxns = ()
#     if unimol:
#         for chi in r1_chis:
#             rxns += automol.reac.enumerate_from_amchis([chi], rxn_type=rxn_type)
#     if bimol:
#         for chi1, chi2 in zip(r1_chis, r2_chis, strict=True):
#             rxns += automol.reac.enumerate_from_amchis([chi1,chi2], rxn_type=rxn_type)

#     # prds = None if prds is None else
#     print(mech)
#     print(rxn_type)
#     print(r1_chis)
#     print(r2_chis)
#     print(p_chis)


# properties
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
        rnames, pnames, _ = data.reac.read_chemkin_equation(eq)
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


# def display_reactions(
#     mech: Mechanism,
#     eqs: Collection | None = None,
#     stereo: bool = True,
#     keys: tuple[str, ...] = (Reactions.eq,),
#     spc_keys: tuple[str, ...] = (Species.smi,),
# ):
#     """Display the reactions in a mechanism.

#     :param mech: _description_
#     :param eqs: _description_, defaults to None
#     :param stereo: _description_, defaults to True
#     :param keys: _description_, defaults to (Reactions.eq,)
#     :param spc_keys: _description_, defaults to (Species.smi,)
#     """
#     pass
