"""Functions for updating mechanisms from CHEMKIN-formatted files."""

from pathlib import Path

from ... import _mech
from ..._mech import Mechanism
from . import read


def thermo(mech: Mechanism, inp: str | Path) -> Mechanism:
    """Update thermochemical data in mechanism.

    :param mech: Mechanism
    :param inp: Thermo file or string
    :return: Mechanism
    """
    spc_df = _mech.species(mech)
    spc_df = read.thermo(inp, spc_df=spc_df)
    return _mech.set_species(mech, spc_df)
