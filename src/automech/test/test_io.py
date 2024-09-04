"""Test automech.io functions."""

import tempfile
from pathlib import Path

import pytest

import automech

DATA_PATH = Path(__file__).parent / "data"
TEMP_PATH = tempfile.gettempdir()


def check_counts(mech, ref_nrxns, ref_nspcs):
    """Check that the reaction and species counts are correct.

    :param mech: A mechanism object
    :param ref_nrxns: The correct reaction count
    :param ref_nspcs: The correct species count
    """
    nrxns = automech.reaction_count(mech)
    nspcs = automech.species_count(mech)
    assert nrxns == ref_nrxns, f"{nrxns} != {ref_nrxns}"
    assert nspcs == ref_nspcs, f"{nspcs} != {ref_nspcs}"


@pytest.mark.parametrize(
    "mech_file_name, nrxns, nspcs",
    [
        ("butane.dat", 101, 76),
        ("ethylene.dat", 26, 31),
    ],
)
def test__chemkin(mech_file_name, nrxns, nspcs):
    """Test automech.io.chemkin."""
    # Read
    mech_path = Path(DATA_PATH) / mech_file_name
    mech = automech.io.chemkin.read.mechanism(mech_path)
    print(mech)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)

    # Write
    out_mech_path = Path(TEMP_PATH) / mech_file_name
    mech_str = automech.io.chemkin.write.mechanism(mech, out=out_mech_path)
    print(mech_str)
    #   - Check the string output
    mech = automech.io.chemkin.read.mechanism(mech_str)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)
    #   - Check the file output
    mech = automech.io.chemkin.read.mechanism(out_mech_path)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)


@pytest.mark.parametrize(
    "rxn_file_name, spc_file_name, nrxns, nspcs",
    [
        ("propyl.dat", "propyl_species.csv", 8, 12),
        ("syngas.dat", "syngas_species.csv", 78, 18),
    ],
)
def test__mechanalyzer(rxn_file_name, spc_file_name, nrxns, nspcs):
    """Test automech.io.mechanalyzer."""
    # Read
    rxn_path = Path(DATA_PATH) / rxn_file_name
    spc_path = Path(DATA_PATH) / spc_file_name
    mech = automech.io.mechanalyzer.read.mechanism(rxn_path, spc_path)
    print(mech)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)


@pytest.mark.parametrize(
    "rxn_file_name, spc_file_name, nrxns, nspcs",
    [
        ("cyclopentene.dat", "cyclopentene_species.txt", 100, 63),
    ],
)
def test__rmg(rxn_file_name, spc_file_name, nrxns, nspcs):
    """Test automech.io.rmg."""
    # Read
    rxn_path = Path(DATA_PATH) / rxn_file_name
    spc_path = Path(DATA_PATH) / spc_file_name
    mech = automech.io.rmg.read.mechanism(rxn_path, spc_path)
    print(mech)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)


if __name__ == "__main__":
    # test__chemkin__read("butane.dat", 101, 76)
    # test__chemkin("LLNL_C2H4_mech.dat", 26, 31)
    # test__mechanalyzer("propyl.dat", "propyl_species.csv", 8, 12)
    test__mechanalyzer("syngas.dat", "syngas_species.csv", 78, 18)
    # test__rmg("cyclopentene.inp", "cyclopentene_species.txt", 100, 63)
