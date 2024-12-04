"""Network mechanism representation."""

from pathlib import Path

import automol
import networkx
import pyvis

from . import data
from .schema import Reaction, Species


# transformation
def pes_components(net: networkx.Graph) -> networkx.Graph:
    """Determine the PES components in a network.

    :param net: A reaction network
    :return: The PES component networks
    """


# serialization
def string(net: networkx.Graph) -> str:
    """Serialize a reaction network as a string.

    :param net: A reaction network
    :return: The string serialization
    """
    return repr(networkx.adjacency_data(net))


# display
def display(
    net: networkx.Graph,
    stereo: bool = True,
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display the mechanism as a reaction network.

    :param net: A reaction network
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param node_exclude_formulas: Formulas for species to be excluded as nodes
    :param out_name: The name of the HTML file for the network visualization
    :param out_dir: The name of the directory for saving the network visualization
    :param open_browser: Whether to open the browser automatically
    """
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
        mech_vis.add_edge(k1, k2, title=data.reac.write_chemkin_equation(rcts, prds))

    # Generate the HTML file
    mech_vis.write_html(str(out_dir / out_name), open_browser=open_browser)
