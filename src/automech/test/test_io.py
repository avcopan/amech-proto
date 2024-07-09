"""Test automech.io functions."""

from pathlib import Path

import automech


def test__io__chemkin(data_directory_path):
    """Test automech.io.chemkin."""
    butane_mech = Path(data_directory_path) / "butane.dat"
    spc_df = automech.io.chemkin.species(butane_mech)
    rxn_df = automech.io.chemkin.reactions(butane_mech)
    print(butane_mech)
    print(rxn_df)
    print(spc_df)
