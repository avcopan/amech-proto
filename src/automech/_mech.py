"""Definition and core functionality of the mechanism data structure."""

import itertools
import tempfile
from pathlib import Path

import automol
import pyvis
from IPython import display as ipd
from pandera.typing import DataFrame, Series
from pydantic import BaseModel

from . import data, schema
from .util import df_


class Mechanism(BaseModel):
    """A chemical kinetic mechanism."""

    reactions: DataFrame[schema.Reactions]
    species: DataFrame[schema.Species]


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


# getters
def species(mech: Mechanism) -> DataFrame[schema.Species]:
    """Get the species dataframe for a mechanism.

    :param mech: The mechanism
    :return: The mechanism's species dataframe
    """
    return mech.species


def reactions(mech: Mechanism) -> DataFrame[schema.Reactions]:
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
        chi = row[schema.Species.chi]
        fml = automol.amchi.formula(chi)
        if any(automol.form.match(fml, f) for f in excl_fmls):
            return True
        return False

    spc_df["excluded"] = spc_df.progress_apply(_is_excluded, axis=1)
    excl_names = list(spc_df[spc_df["excluded"]][schema.Species.name])

    image_dir_path = Path("images")
    image_dir_path.mkdir(exist_ok=True)

    def _create_image(row: Series):
        chi = row[schema.Species.chi]
        gra = automol.amchi.graph(chi, stereo=stereo)
        chk = automol.amchi.amchi_key(chi)
        svg_str = automol.graph.svg_string(gra, image_size=100)

        path = image_dir_path / f"{chk}.svg"
        with open(path, mode="w") as file:
            file.write(svg_str)

        return str(path)

    spc_df["image_path"] = spc_df.progress_apply(_create_image, axis=1)

    net = pyvis.network.Network(directed=True, notebook=True, cdn_resources='in_line')

    def _add_node(row: Series):
        name = row[schema.Species.name]
        smi = row[schema.Species.smi]
        path = row["image_path"]
        if name not in excl_names:
            net.add_node(name, shape="image", image=path, title=smi)

    def _add_edge(row: Series):
        eq = row[schema.Reactions.eq]
        rnames, pnames, _ = data.reac.read_chemkin_equation(eq)
        for rname, pname in itertools.product(rnames, pnames):
            if rname not in excl_names and pname not in excl_names:
                net.add_edge(rname, pname, title=eq)

    spc_df.progress_apply(_add_node, axis=1)
    rxn_df.progress_apply(_add_edge, axis=1)
    ipd.display(net.show(out))


# def display_reactions(
#     mech: Mechanism,
#     eqs: Collection | None = None,
#     stereo: bool = True,
#     keys: tuple[str, ...] = (schema.Reactions.eq,),
#     spc_keys: tuple[str, ...] = (schema.Species.smi,),
# ):
#     """Display the reactions in a mechanism.

#     :param mech: _description_
#     :param eqs: _description_, defaults to None
#     :param stereo: _description_, defaults to True
#     :param keys: _description_, defaults to (schema.Reactions.eq,)
#     :param spc_keys: _description_, defaults to (schema.Species.smi,)
#     """
#     pass
