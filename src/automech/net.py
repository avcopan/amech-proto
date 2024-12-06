"""Network mechanism representation."""

import itertools
from collections.abc import Sequence
from pathlib import Path

import automol
import networkx
import pyvis

from . import data
from .schema import Reaction, Species

COLOR_SEQUENCE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


class Key:
    id = "id_"
    color = "color"
    excluded_species = "excluded_species"
    excluded_reactions = "excluded_reactions"


# transformation
def connected_components(net: networkx.MultiGraph) -> list[networkx.MultiGraph]:
    """Determine the connected components of a network.

    :param net: A reaction network
    :return: The connected components
    """
    return [net.subgraph(ks).copy() for ks in networkx.connected_components(net)]


def pes_components(
    net: networkx.MultiGraph, formula: dict | str | None = None
) -> list[networkx.MultiGraph]:
    """Determine the PES components in a network.

    :param net: A reaction network
    :return: The PES component networks
    """
    pes_nets = pes_networks(net) if formula is None else [pes_network(net, formula)]
    return list(itertools.chain(*map(connected_components, pes_nets)))


def pes_networks(net: networkx.MultiGraph) -> list[networkx.MultiGraph]:
    """Determine the PES networks in a larger reaction network.

    :param net: A reaction network
    :return: The PES component networks
    """
    fmls = automol.form.unique([d[Reaction.formula] for *_, d in net.edges.data()])
    return [pes_network(net, f) for f in fmls]


def pes_network(net: networkx.MultiGraph, formula: dict | str) -> networkx.MultiGraph:
    """Select the network associated with a specific PES.

    :param net: A reaction network
    :param formula: The formula to select
    :return: The PES network
    """
    edge_keys = [
        tuple(k)
        for *k, d in net.edges.data(keys=True)
        if automol.form.equal(d[Reaction.formula], formula)
    ]
    return net.edge_subgraph(edge_keys)


# serialization
def dict_(net: networkx.MultiGraph) -> dict[object, object]:
    """Serialize a reaction network as a string.

    :param net: A reaction network
    :return: The string serialization
    """
    return networkx.adjacency_data(net)


def string(net: networkx.MultiGraph) -> str:
    """Serialize a reaction network as a string.

    :param net: A reaction network
    :return: The string serialization
    """
    return repr(dict_(net))


# display
def display(
    net_: networkx.MultiGraph | Sequence[networkx.MultiGraph],
    stereo: bool = True,
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display the mechanism as a reaction network.

    :param net: A reaction network or sequence or networks
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param node_exclude_formulas: Formulas for species to be excluded as nodes
    :param out_name: The name of the HTML file for the network visualization
    :param out_dir: The name of the directory for saving the network visualization
    :param open_browser: Whether to open the browser automatically
    """
    nets = [net_] if isinstance(net_, networkx.MultiGraph) else net_
    nets = [n.copy() for n in nets]

    # Set different edge colors to distinguish components
    color_cycle = itertools.cycle(COLOR_SEQUENCE)
    for n in nets:
        networkx.set_edge_attributes(n, next(color_cycle), name=Key.color)
    net = networkx.compose_all(nets)

    if not net.nodes:
        print(
            f"The reaction network is empty. Skipping visualization...\n{string(net)}"
        )
        return

    out_dir: Path = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    img_dir = Path("img")
    (out_dir / img_dir).mkdir(exist_ok=True)

    def _image_path(chi):
        """Create an SVG molecule drawing and return the path."""
        gra = automol.amchi.graph(chi, stereo=stereo)
        chk = automol.amchi.amchi_key(chi)
        svg_str = automol.graph.svg_string(gra, image_size=100)

        path = img_dir / f"{chk}.svg"
        with open(out_dir / path, mode="w") as file:
            file.write(svg_str)

        return str(path)

    # Transfer data over to PyVIS
    mech_vis = pyvis.network.Network(
        directed=True, notebook=True, cdn_resources="in_line"
    )
    for k, d in net.nodes.data():
        smi = d[Species.smiles]
        chi = d[Species.amchi]
        mech_vis.add_node(k, title=smi, shape="image", image=_image_path(chi))

    for k1, k2, d in net.edges.data():
        rcts = d[Reaction.reactants]
        prds = d[Reaction.products]
        color = d[Key.color]
        mech_vis.add_edge(
            k1, k2, title=data.reac.write_chemkin_equation(rcts, prds), color=color
        )

    # Generate the HTML file
    mech_vis.write_html(str(out_dir / out_name), open_browser=open_browser)
