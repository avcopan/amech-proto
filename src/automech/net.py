"""Reaction networks."""

import itertools
from collections.abc import Sequence
from pathlib import Path

import automol
import more_itertools as mit
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


# type aliases
Network = networkx.MultiGraph
Node = tuple[str, ...]
Edge = tuple[Node, Node]
Data = dict[str, object]
NodeDatum = tuple[Node, Data]
EdgeDatum = tuple[Node, Node, Data]
MultiEdgeDatum = tuple[Node, Node, int, Data]


# constructors
def from_data(
    node_data: list[Node | NodeDatum],
    edge_data: list[Edge | EdgeDatum | MultiEdgeDatum],
    aux_data: Data | None = None,
) -> Network:
    """Construct a network from data.

    :param node_data: Node data
    :param edge_data: Edge data
    :param aux_data: Auxiliary data
    :return: The network
    """
    aux_data = {} if aux_data is None else aux_data
    net = Network(**aux_data)
    net.add_nodes_from(node_data)
    net.add_edges_from(edge_data)
    return net


def copy(net: Network) -> Network:
    """Get a copy of a network.

    :param net: A network
    :return: A copy of the network
    """
    return from_data(
        node_data=node_data(net), edge_data=edge_data(net), aux_data=auxiliary_data(net)
    )


# getters
def nodes(net: Network) -> list[Node]:
    """Get the list of nodes in the network.

    :param net: A network
    :return: The nodes
    """
    return list(net.nodes)


def edges(net: Network, edges: Sequence[Edge] | None = None) -> list[Edge]:
    """Get the list of edges in the network.

    Includes the keys for each edge.

    :param net: A network
    :param es: Optionally, validate a set of keys
    :return: The nodes
    """

    def _edges(edge: Edge) -> list[Edge]:
        edge = tuple(edge)
        if len(edge) == 3:
            assert edge in net.edges
            return [edge]
        return [(*edge, k) for k in net.get_edge_data(*edge)]

    return (
        list(net.edges)
        if edges is None
        else list(mit.unique_everseen(itertools.chain(*map(_edges, edges))))
    )


def node_data(net: Network) -> list[NodeDatum]:
    """Get the data associated with each node.

    :param net: A network
    :return: The node data
    """
    return list(net.nodes.data())


def edge_data(net: Network) -> list[MultiEdgeDatum]:
    """Get the data associated with each edge.

    :param net: A network
    :return: The edge data
    """
    return list(net.edges.data(keys=True))


def auxiliary_data(net: Network) -> dict[str, object]:
    """Get additional data associated with a network.

    :param net: A network
    :return: The additional data
    """
    return dict(net.graph)


# setters
def set_nodes(net: Network, node_data: list[Node | NodeDatum]) -> Network:
    """Get the list of nodes in the network.

    :param net: A network
    :param node_data: Node data
    :return: The updated network
    """
    return from_data(
        node_data=node_data, edge_data=edge_data(net), aux_data=auxiliary_data(net)
    )


def set_edges(net: Network, edge_data: list[Edge | EdgeDatum]) -> Network:
    """Get the list of edges in the network.

    :param net: A network
    :param edge_data: Edge data
    :return: The updated network
    """
    return from_data(
        node_data=node_data(net), edge_data=edge_data, aux_data=auxiliary_data(net)
    )


def set_auxiliary_data(net: Network, aux_data: dict[str, object]) -> Network:
    """Get additional data associated with a network.

    :param net: A network
    :param aux_data: Auxiliary data
    :return: The updated network
    """
    return from_data(
        node_data=node_data(net), edge_data=edge_data(net), aux_data=aux_data
    )


# properties
def species_names(net: Network) -> list[str]:
    """Get the list of species names in a network.

    Only includes the names of species that are in nodes.

    :param net: A network
    :return: The species names
    """
    return sorted(
        mit.unique_everseen(
            itertools.chain(*([n] if isinstance(n, str) else n for n in nodes(net)))
        )
    )


def node_neighbors(net: Network, node: Node) -> list[Node]:
    """Get the list of neighbors of a node.

    :param net: A network
    :param node: A node in the network
    :return: The list of neighbors
    """
    return list(net[node])


# transformations
def add_node(
    net: Network,
    node: Node,
    data: Data | None = None,
    source_net: Network | None = None,
) -> Network:
    """Add a node (optionally, by pulling data from another network).

    :param net: A network
    :param node: A node to add
    :param data: Node data
    :param source_net: A network to pull node data from
    :return: The updated network
    """
    net = copy(net)
    data = data if data is not None else source_net.nodes[node]
    net.add_node(node, **data)
    return net


def add_edge(
    net: Network,
    edge: Edge,
    key: int | None = None,
    data: Data | None = None,
    node_data: dict[Node, Data] | None = None,
    source_net: Network | None = None,
) -> Network:
    """Add an edge (optionally, by pulling data from another network).

    If `key` is set, this adds a single edge.
    Otherwise, this is assumed to add data for all edges connecting these nodes.

    If pulling data from another network, data for the nodes will be added too.

    :param net: A network
    :param edge: An edge to add
    :param data: Edge data
    :param source_net: A network to pull edge data from
    :return: The updated network
    """
    net = copy(net)

    if source_net is not None:
        node_data = {n: source_net.nodes[n] for n in edge}
        data = source_net.get_edge_data(*edge, key=key)

    data = data if key is None or data is None else {key: data}

    node_data = {} if node_data is None else node_data
    data = {} if data is None else data

    # Add nodes
    for node in edge:
        net.add_node(node, **node_data.get(node, {}))

    # Add edge(s)
    for key_ in data:
        net.add_edge(*edge, key=key_, **data.get(key_, {}))

    return net


def remove_nodes(net: Network, nodes: list[Node]) -> Network:
    """Remove a list of nodes from the network.

    :param net: A network
    :param nodes: Nodes to remove
    :return: The updated network
    """
    net = copy(net)
    net.remove_nodes_from(nodes)
    return net


def remove_edges(net: Network, edges: list[Edge]) -> Network:
    """Remove a list of edges from the network.

    :param net: A network
    :param edges: Edges to remove
    :return: The updated network
    """
    net = copy(net)
    net.remove_edges_from(edges)
    return net


# unions and subgraphs
def union_all(nets: Sequence[Network]) -> Network:
    """Get the union of a sequence of networks.

    :param nets: A sequence of networks
    :return: The combined network
    """
    net = networkx.compose_all(nets)
    return set_auxiliary_data(
        net, aux_data={k: v for n in nets for k, v in auxiliary_data(n).items()}
    )


def union(net1: Network, net2: Network) -> Network:
    """Get the union of two networks.

    :param net1: A network
    :param net2: A network
    :return: The combined network
    """
    return union_all([net1, net2])


def subnetwork(net: Network, ns: Sequence[Node], aux: bool = False) -> Network:
    """Extract a node-induced sub-network from a network.

    :param net: A network
    :param keys: A sequence of node keys
    :param aux: Pass along auxiliary data?
    :return: The network
    """
    sub_net = networkx.subgraph(net, ns)
    return from_data(
        node_data=node_data(sub_net),
        edge_data=edge_data(sub_net),
        aux_data=auxiliary_data(net) if aux else {},
    )


def edge_subnetwork(net: Network, es: Sequence[Edge], aux: bool = False) -> Network:
    """Extract a node-induced sub-network from a network.

    :param net: A network
    :param keys: A sequence of edge keys
    :param aux: Pass along auxiliary data?
    :return: The network
    """
    sub_net = networkx.edge_subgraph(net, edges(net, edges=es))
    return from_data(
        node_data=node_data(sub_net),
        edge_data=edge_data(sub_net),
        aux_data=auxiliary_data(net) if aux else {},
    )


def connected_components(net: Network, aux: bool = False) -> list[Network]:
    """Determine the connected components of a network.

    :param net: A network
    :param aux: Pass along auxiliary data?
    :return: The connected components
    """
    return [subnetwork(net, ks, aux=aux) for ks in networkx.connected_components(net)]


def isolates(net: Network, aux: bool = False) -> Network:
    """Get isolated species as a "network".

    :param net: A network
    :param aux: Pass along auxiliary data?
    :return: The isolated species
    """
    return subnetwork(net, networkx.isolates(net), aux=aux)


def pes_network(net: Network, formula: dict | str, aux: bool = False) -> Network:
    """Select the network associated with a specific PES.

    :param net: A network
    :param formula: The formula to select
    :param aux: Pass along auxiliary data?
    :return: The PES network
    """
    fml = automol.form.from_string(formula) if isinstance(formula, str) else formula
    edge_keys = [
        tuple(k)
        for *k, d in edge_data(net)
        if automol.form.equal(d[Reaction.formula], fml)
    ]
    return edge_subnetwork(net, edge_keys, aux=aux)


def subpes_networks(net: Network, aux: bool = False) -> list[Network]:
    """Determine the PES networks in a larger network.

    :param net: A network
    :param aux: Pass along auxiliary data?
    :return: The PES component networks
    """
    # Set up empty list of sub-PES networks
    nets = []

    # Determine the sub-PES networks among the unimolecular nodes
    multi_nodes = [n for n in nodes(net) if len(n) > 1]
    net_uni = remove_nodes(net, multi_nodes)
    nets.extend(connected_components(net_uni, aux=aux))

    # Partition the missing edges of the non-unimolecular nodes into those connected to
    # unimolecular nodes and those connected to other non-unimolecular nodes
    multi_to_uni_edges = [
        (n1, n2) for n1 in multi_nodes for n2 in node_neighbors(net, n1) if len(n2) == 1
    ]
    multi_to_multi_edges = [
        (n1, n2) for n1 in multi_nodes for n2 in node_neighbors(net, n1) if len(n2) > 1
    ]

    # For each multi-to-unimolecular edge, simply add it to the appropriate sub-PES
    for multi_node, uni_node in multi_to_uni_edges:
        i, x = next((i, x) for i, x in enumerate(nets) if uni_node in nodes(x))
        nets[i] = add_edge(x, (multi_node, uni_node), source_net=net)

    for edge in multi_to_multi_edges:
        i, x = next(
            ((i, x) for i, x in enumerate(nets) if all(n in nodes(x) for n in edge)),
            (None, None),
        )
        if i is None:
            nets.append(edge_subnetwork(net, [edge], aux=True))
        else:
            nets[i] = add_edge(x, edge, source_net=net)

    return nets


# serialization
def dict_(net: Network) -> Data:
    """Serialize a network as a string.

    :param net: A network
    :return: The string serialization
    """
    return networkx.adjacency_data(net)


def string(net: Network) -> str:
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


def display(
    net: Network,
    stereo: bool = True,
    color_subpes: bool = True,
    species_centered: bool = False,
    exclude_formulas: Sequence[str] = DEFAULT_EXCLUDE_FORMULAS,
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display the mechanism as a network.

    :param net: A network
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param color_pes: Add distinct colors to the different PESs
    :param species_centered: Display as a species-centered network?
    :param exclude_formulas: If species-centered, exclude these species from display
    :param out_name: The name of the HTML file for the network visualization
    :param out_dir: The name of the directory for saving the network visualization
    :param open_browser: Whether to open the browser automatically
    """
    if not nodes(net):
        print(f"The network is empty. Skipping visualization...\n{string(net)}")
        return

    out_dir: Path = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Set different edge colors to distinguish components
    if color_subpes:
        color_cycle = itertools.cycle(COLOR_SEQUENCE)
        nets = subpes_networks(net)
        for n in nets:
            networkx.set_edge_attributes(n, next(color_cycle), name=Key.color)
        net = union_all(nets)

    # Convert to species-centered network, if requested
    net = (
        _species_centered_network(net, exclude_formulas=exclude_formulas)
        if species_centered
        else net
    )

    # Transfer data over to PyVIS
    mech_vis = pyvis.network.Network(
        directed=True, notebook=True, cdn_resources="in_line"
    )
    for k, d in node_data(net):
        k = k if isinstance(k, str) else "+".join(k)
        chi = (
            d[Species.amchi]
            if Species.amchi in d
            else automol.amchi.join([s[Species.amchi] for s in d[Key.species]])
        )
        image_path = _image_file_from_amchi(chi, out_dir=out_dir, stereo=stereo)
        mech_vis.add_node(k, shape="image", image=image_path)

    for k1, k2, _, d in edge_data(net):
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


def _species_centered_network(
    net: Network,
    exclude_formulas: Sequence[str] = DEFAULT_EXCLUDE_FORMULAS,
) -> networkx.MultiGraph:
    """Get a species-centered reaction network (nodes are species).

    :param net: A network
    :return: A species-centered network (not usable with the functions above)
    """
    excl_fmls = list(map(automol.form.from_string, exclude_formulas))

    def is_excluded(fml):
        return any(automol.form.match(fml, f) for f in excl_fmls)

    net0 = net

    nodes = []
    node_data = []
    for ks, rd in net0.nodes.data():
        for k, d in zip(ks, rd[Key.species], strict=True):
            fml = d[Species.formula]
            if not is_excluded(fml):
                nodes.append(k)
                node_data.append((k, d))

    edge_data = []
    for k1s, k2s, d in net0.edges.data():
        for k1, k2 in itertools.product(k1s, k2s):
            if k1 in nodes and k2 in nodes:
                edge_data.append((k1, k2, d))

    net = Network()
    net.add_nodes_from(node_data)
    net.add_edges_from(edge_data)
    return net


def _image_file_from_amchi(chi, out_dir: str | Path, stereo: bool = True):
    """Create an SVG molecule drawing and return the path."""
    out_dir = Path(out_dir)
    img_dir = Path("img")
    (out_dir / img_dir).mkdir(exist_ok=True)

    gra = automol.amchi.graph(chi, stereo=stereo)
    svg_str = automol.graph.svg_string(gra, image_size=100)

    chk = automol.amchi.amchi_key(chi)
    path = img_dir / f"{chk}.svg"
    with open(out_dir / path, mode="w") as file:
        file.write(svg_str)

    return str(path)
