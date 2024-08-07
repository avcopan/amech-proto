"""Test automech.io functions."""

from pathlib import Path

import automech


def test__io__chemkin(data_directory_path):
    """Test automech.io.chemkin."""
    mech_path = Path(data_directory_path) / "butane.dat"
    mech = automech.io.chemkin.read.mechanism(mech_path)
    print(mech_path)
    print(mech)


def test__io__mechanalyzer(data_directory_path):
    """Test automech.io.mechanalyzer."""
    rxn_path = Path(data_directory_path) / "propyl.dat"
    spc_path = Path(data_directory_path) / "propyl_species.csv"
    mech = automech.io.mechanalyzer.read.mechanism(rxn_path, spc_path)
    print(rxn_path)
    print(spc_path)
    print(mech)


def test__io__rmg(data_directory_path):
    """Test automech.io.rmg."""
    rxn_path = Path(data_directory_path) / "cyclopentene.inp"
    spc_path = Path(data_directory_path) / "cyclopentene_species.txt"
    mech = automech.io.rmg.read.mechanism(rxn_path, spc_path)
    print(rxn_path)
    print(spc_path)
    print(mech)


if __name__ == "__main__":
    test__io__chemkin("/home/avcopan/code/amech-proto/src/automech/test/data")
    test__io__mechanalyzer("/home/avcopan/code/amech-proto/src/automech/test/data")
    test__io__rmg("/home/avcopan/code/amech-proto/src/automech/test/data")
