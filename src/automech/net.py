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


# properties
def species_names(net: networkx.MultiGraph) -> list[str]:
    """Get the species names in a network.

    :param net: A network
    :return: The species names in this network
    """
    return list(net.nodes)


# operations
def union(net1: networkx.MultiGraph, net2: networkx.MultiGraph) -> networkx.MultiGraph:
    """Get the union of two networks.

    :param net1: A network
    :param net2: A network
    :return: The combined network
    """
    return union_all([net1, net2])


def union_all(nets: Sequence[networkx.MultiGraph]) -> networkx.MultiGraph:
    """Get the union of a sequence of networks.

    :param nets: A sequence of networks
    :return: The combined network
    """
    net: networkx.MultiGraph = networkx.compose_all(nets)
    net.graph = {k: v for n in nets for k, v in dict(n.graph).items()}
    return net


def subnetwork(net: networkx.MultiGraph, keys: Sequence[str]) -> networkx.MultiGraph:
    """Extract a node-induced sub-network from a network.

    :param net: A network
    :param keys: A sequence of node keys
    :return: The network
    """
    sub_net: networkx.MultiGraph = networkx.subgraph(net, keys).copy()
    sub_net.graph = dict(net.graph).copy()
    return sub_net


def edge_subnetwork(
    net: networkx.MultiGraph, keys: Sequence[tuple[str, str]]
) -> networkx.MultiGraph:
    """Extract a node-induced sub-network from a network.

    :param net: A network
    :param keys: A sequence of edge keys
    :return: The network
    """
    sub_net: networkx.MultiGraph = networkx.edge_subgraph(net, keys).copy()
    sub_net.graph = dict(net.graph).copy()
    return sub_net


# transformation
def connected_components(net: networkx.MultiGraph) -> list[networkx.MultiGraph]:
    """Determine the connected components of a network.

    :param net: A network
    :return: The connected components
    """
    sub_nets = [subnetwork(net, ks) for ks in networkx.connected_components(net)]
    for sub_net in sub_nets:
        sub_net.graph = dict(net.graph).copy()
    return sub_nets


def combined_subpes_networks(
    net: networkx.MultiGraph,
    formula: dict | str | None = None,
    spc_names: Sequence[str] | None = None,
) -> list[networkx.MultiGraph]:
    """Determine the connected sub-PESs in a network.

    :param net: A network
    :param formula: The formula to select
    :param spc_names: Optionally, only include networks containing these species
    :return: The connected sub-PES networks
    """
    return union_all(subpes_networks(net, formula=formula, spc_names=spc_names))


def subpes_networks(
    net: networkx.MultiGraph,
    formula: dict | str | None = None,
    spc_names: Sequence[str] | None = None,
) -> list[networkx.MultiGraph]:
    """Determine the connected sub-PESs in a network.

    :param net: A network
    :param formula: The formula to select
    :param spc_names: Optionally, only include networks containing these species
    :return: The connected sub-PES networks
    """
    pes_nets = pes_networks(net) if formula is None else [pes_network(net, formula)]
    subpes_nets = list(itertools.chain(*map(connected_components, pes_nets)))
    if spc_names is not None:
        subpes_nets = [n for n in subpes_nets if any(s in n for s in spc_names)]
    return subpes_nets


def isolates(net: networkx.MultiGraph) -> networkx.MultiGraph:
    """Get isolated species as a "network".

    :param net: A network
    :return: The isolated species
    """
    return subnetwork(net, networkx.isolates(net))


def pes_network(net: networkx.MultiGraph, formula: dict | str) -> networkx.MultiGraph:
    """Select the network associated with a specific PES.

    :param net: A network
    :param formula: The formula to select
    :return: The PES network
    """
    fml = automol.form.from_string(formula) if isinstance(formula, str) else formula
    edge_keys = [
        tuple(k)
        for *k, d in net.edges.data(keys=True)
        if automol.form.equal(d[Reaction.formula], fml)
    ]
    return edge_subnetwork(net, edge_keys)


def pes_networks(
    net: networkx.MultiGraph, with_isolates: bool = False
) -> list[networkx.MultiGraph]:
    """Determine the PES networks in a larger network.

    :param net: A network
    :param with_isolates: Whether to include isolated species as a "network"
    :return: The PES component networks
    """
    fmls = automol.form.unique([d[Reaction.formula] for *_, d in net.edges.data()])
    nets = [pes_network(net, f) for f in fmls]
    if with_isolates:
        nets.append(isolates(net))
    return nets


def extend_subnetwork_along_pes(
    net: networkx.MultiGraph,
    sub_net: networkx.MultiGraph,
    formula: dict | str | None = None,
    radius: int = 1,
) -> networkx.MultiGraph:
    """Extend a sub-network along a specific PES.

    :param net: A network
    :param sub_net: A sub-network
    :param formula: The formula to select
    :return: The extended sub-network
    """
    pes_net = pes_network(net, formula=formula)
    spc_names = [n for n in species_names(sub_net) if n in pes_net]
    ext_net = neighborhood(pes_net, spc_names=spc_names, radius=radius)
    return union(sub_net, ext_net)


def neighborhood(
    net: networkx.MultiGraph,
    spc_names: Sequence[str],
    radius: int = 1,
) -> networkx.MultiGraph:
    """Determine the neighborhood of a set of species.

    :param net: A network
    :param spc_names: A list of species names
    :param radius: Maximum distance of neighbors to include
    :param pes_extend: Extend the neighborhood to include full connected sub-PESs
    :return: The nth-order neighborhood network
    """
    nei_nets = [
        networkx.ego_graph(net, n, radius=radius, undirected=True) for n in spc_names
    ]
    for nei_net_ in nei_nets:
        nei_net_.graph = dict(net.graph).copy()
    nei_net = union_all(nei_nets)
    nei_net.graph[Key.excluded_reactions] = []
    return nei_net


# serialization
def dict_(net: networkx.MultiGraph) -> dict[object, object]:
    """Serialize a network as a string.

    :param net: A network
    :return: The string serialization
    """
    return networkx.adjacency_data(net)


def string(net: networkx.MultiGraph) -> str:
    """Serialize a network as a string.

    :param net: A network
    :return: The string serialization
    """
    return repr(dict_(net))


# display
def display(
    net: networkx.MultiGraph,
    stereo: bool = True,
    color_pes: bool = True,
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display the mechanism as a network.

    :param net: A network or sequence or networks
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param color_pes: Add distinct colors to the different PESs
    :param node_exclude_formulas: Formulas for species to be excluded as nodes
    :param out_name: The name of the HTML file for the network visualization
    :param out_dir: The name of the directory for saving the network visualization
    :param open_browser: Whether to open the browser automatically
    """
    if not net.nodes:
        print(f"The network is empty. Skipping visualization...\n{string(net)}")
        return

    # Set different edge colors to distinguish components
    if color_pes:
        color_cycle = itertools.cycle(COLOR_SEQUENCE)
        nets = pes_networks(net, with_isolates=True)
        for n in nets:
            networkx.set_edge_attributes(n, next(color_cycle), name=Key.color)
        net = union_all(nets)

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
