"""Reaction networks."""

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

    out_dir: Path = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    img_dir = Path("img")
    (out_dir / img_dir).mkdir(exist_ok=True)

    def _image_file(chis):
        """Create an SVG molecule drawing and return the path."""
        chi = automol.amchi.join(chis)
        gra = automol.amchi.graph(chi)
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
        vis_id = "+".join(k)
        chis = [s[Species.amchi] for s in d[Key.species]]
        mech_vis.add_node(vis_id, shape="image", image=_image_file(chis))

    for k1, k2, d in net.edges.data():
        vis_id1 = "+".join(k1)
        vis_id2 = "+".join(k2)
        rcts = d[Reaction.reactants]
        prds = d[Reaction.products]
        mech_vis.add_edge(
            vis_id1, vis_id2, title=data.reac.write_chemkin_equation(rcts, prds)
        )

    # Generate the HTML file
    mech_vis.write_html(str(out_dir / out_name), open_browser=open_browser)
