"""Test automech.io functions."""

import tempfile
from pathlib import Path

import pytest

import automech

DATA_PATH = Path(__file__).parent / "data"
TEMP_PATH = tempfile.gettempdir()


def check_counts(mech, ref_rcount, ref_scount):
    """Check that the reaction and species counts are correct.

    :param mech: A mechanism object
    :param ref_rcount: The correct reaction count
    :param ref_scount: The correct species count
    """
    rcount = automech.reaction_count(mech)
    scount = automech.species_count(mech)
    assert rcount == ref_rcount, f"{rcount} != {ref_rcount}"
    assert scount == ref_scount, f"{scount} != {ref_scount}"


@pytest.mark.parametrize(
    "mech_file_name,rcount,scount",
    [
        ("butane.dat", 101, 76),
        ("LLNL_C2H4_mech.dat", 26, 31),
    ],
)
def test__chemkin(mech_file_name, rcount, scount):
    """Test automech.io.chemkin.read."""
    # Read
    mech_path = Path(DATA_PATH) / mech_file_name
    mech = automech.io.chemkin.read.mechanism(mech_path)
    print(mech)
    check_counts(mech, ref_rcount=rcount, ref_scount=scount)

    # Write
    out_mech_path = Path(TEMP_PATH) / mech_file_name
    mech_str = automech.io.chemkin.write.mechanism(mech, out=out_mech_path)
    print(mech_str)
    #   - Check the string output
    mech = automech.io.chemkin.read.mechanism(mech_str)
    check_counts(mech, ref_rcount=rcount, ref_scount=scount)
    #   - Check the file output
    mech = automech.io.chemkin.read.mechanism(out_mech_path)
    check_counts(mech, ref_rcount=rcount, ref_scount=scount)


def test__mechanalyzer__read():
    """Test automech.io.mechanalyzer."""
    rxn_path = Path(DATA_PATH) / "propyl.dat"
    spc_path = Path(DATA_PATH) / "propyl_species.csv"
    mech = automech.io.mechanalyzer.read.mechanism(rxn_path, spc_path)
    print(rxn_path)
    print(spc_path)
    print(mech)


def test__rmg__read():
    """Test automech.io.rmg."""
    rxn_path = Path(DATA_PATH) / "cyclopentene.inp"
    spc_path = Path(DATA_PATH) / "cyclopentene_species.txt"
    mech = automech.io.rmg.read.mechanism(rxn_path, spc_path)
    print(rxn_path)
    print(spc_path)
    print(mech)


if __name__ == "__main__":
    # test__chemkin__read("butane.dat", 101, 76)
    test__chemkin("LLNL_C2H4_mech.dat", 26, 31)
    # test__mechanalyzer__read("/home/avcopan/code/amech-proto/src/automech/test/data")
    # test__rmg__read("/home/avcopan/code/amech-proto/src/automech/test/data")
