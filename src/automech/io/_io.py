"""Functions for reading and writing AutoMech-formatted files."""

from pathlib import Path

import polars

from .. import _mech
from .._mech import Mechanism


def write(mech: Mechanism, path: str | Path, prefix: str = "mech") -> None:
    """Write a mechanism to JSON format.

    :param mech: A mechanism
    :param path: The path to write to (either directory or reactions file)
    :param prefix: File name prefix, used if path is a directory
    """
    path = Path(path)
    if not path.is_dir():
        prefix = path.stem.removesuffix("_reactions")
        path = path.parent

    _mech.reactions(mech).write_ndjson(path / f"{prefix}_reactions.json")
    _mech.species(mech).write_ndjson(path / f"{prefix}_species.json")


def read(path: str | Path, prefix: str = "mech") -> Mechanism:
    """Read a mechanism from JSON format.

    :param path: The path to write to (either directory or reactions file)
    :param prefix: File name prefix, used if path is a directory
    :return: The mechanism
    """
    path = Path(path)
    if not path.is_dir():
        prefix = path.stem.removesuffix("_reactions")
        path = path.parent

    return _mech.from_data(
        rxn_inp=polars.read_ndjson(path / f"{prefix}_reactions.json"),
        spc_inp=polars.read_ndjson(path / f"{prefix}_species.json"),
    )
