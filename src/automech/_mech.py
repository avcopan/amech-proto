"""Definition and core functionality of the mechanism data structure."""

import itertools
import textwrap
from collections.abc import Sequence
from pathlib import Path

import automol
import pandas
import pyvis
from IPython import display as ipd
from pandera.typing import DataFrame, Series
from pydantic import BaseModel

from . import data, schema
from .schema import Reactions, Species
from .util import df_


class Mechanism(BaseModel):
    """A chemical kinetic mechanism."""

    reactions: DataFrame[Reactions]
    species: DataFrame[Species]

    def __repr__(self):
        rxn_df_rep = textwrap.indent(repr(self.reactions), "  ")
        spc_df_rep = textwrap.indent(repr(self.species), "  ")
        rxn_rep = textwrap.indent(f"reactions=DataFrame(\n{rxn_df_rep}\n)", "  ")
        spc_rep = textwrap.indent(f"species=DataFrame(\n{spc_df_rep}\n)", "  ")
        return f"Mechanism(\n{rxn_rep},\n{spc_rep},\n)"


# constructors
def from_data(inp, spc_inp, validate: bool = True, smi: bool = False) -> Mechanism:
    """Contruct a mechanism object from data.

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param validate: Validate the data?
    :param smi: Add SMILES column, if missing? (Takes time)
    :return: The mechanism object
    """
    rxn_df = df_.from_csv(inp) if isinstance(inp, str) else inp
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp

    if validate:
        rxn_df = schema.validate_reactions(rxn_df)
        spc_df = schema.validate_species(spc_df, smi=smi)

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
    ids = list(zip(smis, chis, strict=True))
    spc_df = pandas.DataFrame(
        data={
            Species.name: [
                name_dct[s] if s in name_dct else automol.amchi.chemkin_name(c)
                for s, c in ids
            ],
            Species.spin: [
                spin_dct[s] if s in spin_dct else automol.amchi.guess_spin(c)
                for s, c in ids
            ],
            Species.charge: [charge_dct[s] if s in charge_dct else 0 for s, c in ids],
            Species.smi: smis,
            Species.chi: chis,
        }
    )

    # Build the reactions dataframe
    trans_dct = df_.lookup_dict(spc_df, Species.smi, Species.name)
    rxn_smis_lst = list(map(automol.smiles.reaction_reactants_and_products, rxn_smis))
    eqs = [
        data.reac.write_chemkin_equation(rs, ps, trans_dct=trans_dct)
        for rs, ps in rxn_smis_lst
    ]
    rxn_df = pandas.DataFrame(data={Reactions.eq: eqs})
    return from_data(rxn_df, spc_df, validate=True)


# getters
def species(mech: Mechanism) -> DataFrame[Species]:
    """Get the species dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism's species dataframe
    """
    return mech.species


def reactions(mech: Mechanism) -> DataFrame[Reactions]:
    """Get the reactions dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism's reactions dataframe
    """
    return mech.reactions


# properties
def display(
    mech: Mechanism,
    stereo: bool = True,
    exclude: tuple[str, ...] = ("H*", "OH*", "O2H*", "CH*"),
    out: str = "net.html",
) -> None:
    """Display the mechanism as a reaction network.

    :param mech: The mechanism
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param exclude: Formula strings of molecules to exclude from the  network nodes,
        using * for wildcard stoichiometry, defaults to ("H*", "OH*", "O2H*", "CH*")
    :param out: The name of the HTML file to write the network visualization to
    """
    # Read in the mechanism data
    spc_df = species(mech)
    rxn_df = reactions(mech)

    # Handle excluded species
    excl_fmls = tuple(map(automol.form.from_string, exclude))

    def _is_excluded(row: Series):
        chi = row[Species.chi]
        fml = automol.amchi.formula(chi)
        if any(automol.form.match(fml, f) for f in excl_fmls):
            return True
        return False

    spc_df["excluded"] = spc_df.progress_apply(_is_excluded, axis=1)
    excl_names = list(spc_df[spc_df["excluded"]][Species.name])

    image_dir_path = Path("img")
    image_dir_path.mkdir(exist_ok=True)

    def _create_image(row: Series):
        chi = row[Species.chi]
        gra = automol.amchi.graph(chi, stereo=stereo)
        chk = automol.amchi.amchi_key(chi)
        svg_str = automol.graph.svg_string(gra, image_size=100)

        path = image_dir_path / f"{chk}.svg"
        with open(path, mode="w") as file:
            file.write(svg_str)

        return str(path)

    spc_df["image_path"] = spc_df.progress_apply(_create_image, axis=1)

    net = pyvis.network.Network(directed=True, notebook=True, cdn_resources="in_line")

    def _add_node(row: Series):
        name = row[Species.name]
        smi = row[Species.smi]
        path = row["image_path"]
        if name not in excl_names:
            net.add_node(name, shape="image", image=path, title=smi)

    def _add_edge(row: Series):
        eq = row[Reactions.eq]
        rnames, pnames, _ = data.reac.read_chemkin_equation(eq)
        for rname, pname in itertools.product(rnames, pnames):
            if rname not in excl_names and pname not in excl_names:
                net.add_edge(rname, pname, title=eq)

    spc_df.progress_apply(_add_node, axis=1)
    rxn_df.progress_apply(_add_edge, axis=1)
    net.write_html(out, open_browser=True)


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
