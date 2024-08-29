"""Test automech functions."""

import pytest

import automech
from automech.schema import Species


def test__from_smiles():
    """Test automech.from_smiles.

    Also tests the display function, to make sure it runs.
    """
    # Example 1
    mech = automech.from_smiles(
        rxn_smis=["CCC.[OH]>>CC[CH2].O"],
        name_dct={"CCC": "C3H8", "[OH]": "OH", "CC[CH2]": "C3H7y1"},
    )
    automech.display(mech, open_browser=False)
    automech.display_reactions(mech, eqs=["C3H8+OH=C3H7y1+H2O"])

    # Check that the names were correctly applied
    ref_name_dct = {
        "AMChI=1/C3H8/c1-3-2/h3H2,1-2H3": "C3H8",
        "AMChI=1/HO/h1H": "OH",
        "AMChI=1/C3H7/c1-3-2/h1,3H2,2H3": "C3H7y1",
        "AMChI=1/H2O/h1H2": "H2O",
    }
    name_dct = automech.util.df_.lookup_dict(
        automech.species(mech), Species.amchi, Species.name
    )
    assert name_dct == ref_name_dct, f"{name_dct} != {ref_name_dct}"

    # Example 2
    mech = automech.from_smiles(spc_smis=["CCC"], rxn_smis=[])
    print(mech)
    automech.display(mech, open_browser=False)

    # Example 3 (empty mechanism)
    mech = automech.from_smiles()
    print(mech)
    automech.display(mech, open_browser=False)


@pytest.mark.parametrize(
    "rxn_smis,ref_rcount,ref_scount,ref_err_rcount,ref_err_scount",
    [
        (["FC=CF.[OH]>>F[C]=CF.O"], 2, 6, 0, 0),
    ],
)
def test__expand_stereo(
    rxn_smis, ref_rcount, ref_scount, ref_err_rcount, ref_err_scount
):
    """Test automech.expand_stereo."""
    mech = automech.from_smiles(rxn_smis=rxn_smis)
    mech, err_mech = automech.expand_stereo(mech)
    print(mech)
    print(err_mech)
    rcount = automech.reaction_count(mech)
    scount = automech.species_count(mech)
    err_rcount = automech.reaction_count(err_mech)
    err_scount = automech.species_count(err_mech)
    assert rcount == ref_rcount, f"{rcount} != {ref_rcount}"
    assert scount == ref_scount, f"{scount} != {ref_scount}"
    assert err_rcount == ref_err_rcount, f"{err_rcount} != {ref_err_rcount}"
    assert err_scount == ref_err_scount, f"{err_scount} != {ref_err_scount}"


if __name__ == "__main__":
    # test__from_smiles()
    test__expand_stereo(["FC=CF.[OH]>>F[C]=CF.O"], 2, 6, 0, 0)
