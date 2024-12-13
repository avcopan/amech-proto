"""Reaction networks."""

import itertools
from collections.abc import Sequence
from pathlib import Path

import automol
import networkx
import pyvis

from . import data
from .schema import Reaction, Species


class Key:
    # Shared:
    id = "id_"
    formula = "formula"
    # Nodes only:
    species = "species"
    # Edges only:
    color = "color"
    # Graph:
    excluded_species = "excluded_species"
    excluded_reactions = "excluded_reactions"


# unions and subgraphs
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


def connected_components(net: networkx.MultiGraph) -> list[networkx.MultiGraph]:
    """Determine the connected components of a network.

    :param net: A network
    :return: The connected components
    """
    sub_nets = [subnetwork(net, ks) for ks in networkx.connected_components(net)]
    for sub_net in sub_nets:
        sub_net.graph = dict(net.graph).copy()
    return sub_nets


# properties
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
DEFAULT_EXCLUDE_FORMULAS = ("H*O*", "CH*")
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


def species_centered_network(
    net: networkx.MultiGraph,
    exclude_formulas: Sequence[str] = DEFAULT_EXCLUDE_FORMULAS,
) -> networkx.MultiGraph:
    """Get a species-centered reaction network (nodes are species).

    :param net: A network
    :return: A network in which the nodes are individual apecies
    """
    excl_fmls = list(map(automol.form.from_string, exclude_formulas))

    def is_excluded(fml):
        return any(automol.form.match(fml, f) for f in excl_fmls)

    net0 = net

    seen_ids = []
    node_data = []
    excl_ks = []
    excl_spcs = []
    for ks, rd in net0.nodes.data():
        for k, d in zip(ks, rd[Key.species], strict=True):
            id_ = d[Key.id]
            fml = d[Species.formula]
            if id_ not in seen_ids:
                if is_excluded(fml):
                    excl_ks.append(k)
                    excl_spcs.append(d)
                else:
                    node_data.append((k, d))

    edge_data = []
    excl_rxns = []
    for k1s, k2s, d in net0.edges.data():
        if any(all(k in excl_ks for k in ks) for ks in (k1s, k2s)):
            excl_rxns.append(d)
        for k1, k2 in itertools.product(k1s, k2s):
            if k1 not in excl_ks and k2 not in excl_ks:
                edge_data.append((k1, k2, d))

    net = networkx.MultiGraph(
        **{Key.excluded_species: excl_spcs, Key.excluded_reactions: excl_rxns}
    )
    net.add_nodes_from(node_data)
    net.add_edges_from(edge_data)
    return net


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

    def _image_file(chi):
        """Create an SVG molecule drawing and return the path."""
        gra = automol.amchi.graph(chi, stereo=stereo)
        svg_str = automol.graph.svg_string(gra, image_size=100)

        chk = automol.amchi.amchi_key(chi)
        path = img_dir / f"{chk}.svg"
        with open(out_dir / path, mode="w") as file:
            file.write(svg_str)

        return str(path)

    # Transfer data over to PyVIS
    mech_vis = pyvis.network.Network(
        directed=True, notebook=True, cdn_resources="in_line"
    )
    for k, d in net.nodes.data():
        k = k if isinstance(k, str) else "+".join(k)
        chi = (
            d[Species.amchi]
            if Species.amchi in d
            else automol.amchi.join([s[Species.amchi] for s in d[Key.species]])
        )
        mech_vis.add_node(k, shape="image", image=_image_file(chi))

    for k1, k2, d in net.edges.data():
        k1 = k1 if isinstance(k1, str) else "+".join(k1)
        k2 = k2 if isinstance(k2, str) else "+".join(k2)
        rcts = d[Reaction.reactants]
        prds = d[Reaction.products]
        color = d.get(Key.color)
        mech_vis.add_edge(
            k1, k2, title=data.reac.write_chemkin_equation(rcts, prds), color=color
        )

    # Generate the HTML file
    mech_vis.write_html(str(out_dir / out_name), open_browser=open_browser)
