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
    "mech_file_name, nrxns, nspcs, roundtrip",
    [
        ("butane.dat", 101, 76, False),
        ("ethylene.dat", 26, 31, False),
        ("webb_sample.inp", 11, 16, True),
    ],
)
def test__chemkin(mech_file_name, nrxns, nspcs, roundtrip):
    """Test automech.io.chemkin."""
    # Read
    mech_path = Path(DATA_PATH) / mech_file_name
    mech0 = automech.io.chemkin.read.mechanism(mech_path)
    print(mech0)
    check_counts(mech0, ref_nrxns=nrxns, ref_nspcs=nspcs)

    # Write
    out = Path(TEMP_PATH) / mech_file_name
    mech_str = automech.io.chemkin.write.mechanism(mech0, out=out)
    print(mech_str)
    #   - Check the direct output
    mech = automech.io.chemkin.read.mechanism(mech_str)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)
    #   - Check the file output
    mech = automech.io.chemkin.read.mechanism(out)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)

    if roundtrip:
        print("Not yet checking roundtrip equivalence")

    # # Check that the written mechanism has the same content
    # # (Not yet working -- requires loosening of float comparison in `are_equivalent`)
    # if roundtrip:
    #     assert automech.are_equivalent(mech, mech0), f"{mech} != {mech0}"


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

    # Write
    rxn_out = Path(TEMP_PATH) / rxn_file_name
    spc_out = Path(TEMP_PATH) / spc_file_name
    mech_str, csv_str = automech.io.mechanalyzer.write.mechanism(
        mech, rxn_out=rxn_out, spc_out=spc_out, string=True
    )
    print(mech_str)
    #   - Check the direct output
    mech = automech.io.mechanalyzer.read.mechanism(mech_str, csv_str)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)
    #   - Check the file output
    mech = automech.io.mechanalyzer.read.mechanism(rxn_out, spc_out)
    check_counts(mech, ref_nrxns=nrxns, ref_nspcs=nspcs)


@pytest.mark.parametrize(
    "rxn_file_name, spc_file_name, nrxns, nspcs",
    [
        ("cyclopentene.dat", "cyclopentene_species.txt", 100, 63),
        ("webb_sample.inp", "webb_sample_species.txt", 11, 16),
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
    test__chemkin("webb_sample.inp", 11, 16, True)
    # test__chemkin("LLNL_C2H4_mech.dat", 26, 31)
    # test__mechanalyzer("propyl.dat", "propyl_species.csv", 8, 12)
    # test__mechanalyzer("syngas.dat", "syngas_species.csv", 78, 18)
    # test__rmg("cyclopentene.inp", "cyclopentene_species.txt", 100, 63)
