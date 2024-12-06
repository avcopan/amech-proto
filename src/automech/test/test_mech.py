"""Test automech functions."""

from pathlib import Path

import pytest

import automech
from automech.schema import Species

DATA_PATH = Path(__file__).parent / "data"

MECH_EMPTY = automech.from_smiles(spc_smis=[], rxn_smis=[])
MECH_NO_REACIONS = automech.from_smiles(
    spc_smis=["CC=CC"], rxn_smis=[], name_dct={"CC=CC": "C4e2"}
)
MECH_PROPANE = automech.from_smiles(
    rxn_smis=["CCC.[OH]>>CC[CH2].O"],
    name_dct={"CCC": "C3", "[OH]": "OH", "CC[CH2]": "C3y1"},
)
MECH_BUTENE = automech.from_smiles(
    rxn_smis=[
        "CC=CC.[OH]>>CC=C[CH2].O",
        "CC=CC.[OH]>>C[CH]C(O)C",
        "CC=CC.[O]>>CC1C(O1)C",
    ],
    name_dct={"CC=CC": "C4e2", "CC=C[CH2]": "C4e2y1", "CC1C(O1)C": "C4x23"},
)
MECH_BUTENE_SUBSET = automech.from_smiles(
    rxn_smis=[
        "O.CC=C[CH2]>>[OH].CC=CC",
    ],
    name_dct={
        "CC=CC": "C4e2",
        "CC=C[CH2]": "C4e2y1",
    },
)
MECH_BUTENE_ALTERNATIVE_NAMES = automech.from_smiles(
    spc_smis=["CC=CC", "CC=C[CH2]", "O", "[OH]"],
    name_dct={
        "CC=CC": "2-butene",
        "CC=C[CH2]": "1-methylallyl",
        "O": "water",
        "[OH]": "hydroxyl",
    },
)
MECH_BUTENE_WITH_EXCLUDED_REACTIONS = automech.from_smiles(
    rxn_smis=[
        "CC=CC.[OH]>>CC=C[CH2].O",
        "CC=CC.[OH]>>C[CH]C(O)C",
        "CC=CC.[O]>>CC1C(O1)C",
        "[OH].[OH]>>OO",
    ],
    name_dct={"CC=CC": "C4e2", "CC=C[CH2]": "C4e2y1", "CC1C(O1)C": "C4x23"},
)


@pytest.mark.parametrize(
    "mech0",
    [
        MECH_EMPTY,
        MECH_NO_REACIONS,
        MECH_PROPANE,
        MECH_BUTENE,
        MECH_BUTENE_WITH_EXCLUDED_REACTIONS,
    ],
)
def test__network(mech0):
    """Test automech.network."""
    mech = automech.from_network(automech.network(mech0))
    print(mech)
    assert automech.species_count(mech0) == automech.species_count(mech)
    assert automech.reaction_count(mech0) == automech.reaction_count(mech)


@pytest.mark.parametrize(
    "mech, smis, eqs",
    [
        (MECH_EMPTY, None, None),
        (MECH_NO_REACIONS, None, None),
        (MECH_PROPANE, ("CCC", "[OH]"), ("C3+OH=C3y1+H2O",)),
        (MECH_BUTENE, ("CC=CC", "CC=C[CH2]"), ("C4e2+OH=C4e2y1+H2O",)),
    ],
)
def test__display(mech, smis, eqs):
    """Test automech.display."""
    automech.display(mech, open_browser=False)
    automech.display_species(mech, sel_vals=smis, sel_key=Species.smiles)
    automech.display_reactions(mech, eqs=eqs)


@pytest.mark.parametrize(
    "mech, ref_rcount, ref_scount, ref_err_rcount, ref_err_scount, drop_unused",
    [
        (MECH_NO_REACIONS, 0, 2, 0, 2, False),
        (MECH_BUTENE, 6, 8, 1, 3, True),
    ],
)
def test__expand_stereo(
    mech, ref_rcount, ref_scount, ref_err_rcount, ref_err_scount, drop_unused
):
    """Test automech.expand_stereo."""
    exp_mech, err_mech = automech.expand_stereo(mech)
    if drop_unused:
        exp_mech = automech.without_unused_species(exp_mech)
        err_mech = automech.without_unused_species(err_mech)
    print(exp_mech)
    print(err_mech)
    rcount = automech.reaction_count(exp_mech)
    scount = automech.species_count(exp_mech)
    err_rcount = automech.reaction_count(err_mech)
    err_scount = automech.species_count(err_mech)
    assert rcount == ref_rcount, f"{rcount} != {ref_rcount}"
    assert scount == ref_scount, f"{scount} != {ref_scount}"
    assert err_rcount == ref_err_rcount, f"{err_rcount} != {ref_err_rcount}"
    assert err_scount == ref_err_scount, f"{err_scount} != {ref_err_scount}"


@pytest.mark.parametrize(
    "mech0, name_mech, nspcs",
    [
        (MECH_BUTENE, MECH_BUTENE_ALTERNATIVE_NAMES, 4),
    ],
)
def test__rename(mech0, name_mech, nspcs):
    name_dct, missing_names = automech.rename_dict(mech0, name_mech)
    print(name_dct)
    print(missing_names)
    mech = automech.rename(mech0, name_dct)
    mech_drop = automech.rename(mech0, name_dct, drop_missing=True)

    print(mech)
    print(mech_drop)
    assert len(name_dct) + len(missing_names) == automech.species_count(mech0)
    assert automech.species_count(mech) == automech.species_count(mech0)
    assert automech.species_count(mech_drop) == nspcs


@pytest.mark.parametrize(
    "par_mech, mech, rcount, scount",
    [(MECH_BUTENE, MECH_NO_REACIONS, 6, 8)],
)
def test__expand_parent_stereo(par_mech, mech, rcount, scount):
    exp_mech, _ = automech.expand_stereo(mech)
    exp_par_mech = automech.expand_parent_stereo(
        par_mech=par_mech, exp_sub_mech=exp_mech
    )
    print(exp_par_mech)
    assert automech.reaction_count(exp_par_mech) == rcount
    assert automech.species_count(exp_par_mech) == scount


@pytest.mark.parametrize(
    "par_mech0, mech, rcount, scount",
    [(MECH_BUTENE, MECH_BUTENE_SUBSET, 6, 9)],
)
def test__update_parent_reaction_data(par_mech0, mech, rcount, scount):
    exp_mech, _ = automech.expand_stereo(mech)
    par_mech = automech.update_parent_reaction_data(par_mech0, exp_mech)
    print(par_mech)
    assert automech.reaction_count(par_mech) == rcount
    assert automech.species_count(par_mech) == scount


if __name__ == "__main__":
    # test__from_smiles()
    # test__expand_stereo(MECH_BUTENE, 6, 8, 1, 3, True)
    # test__expand_stereo(MECH_NO_REACIONS, 0, 2, 0, 2, False)
    # test__expand_parent_stereo(MECH_BUTENE, MECH_NO_REACIONS, 6, 8)
    # test__rename(MECH_BUTENE, MECH_BUTENE_ALTERNATIVE_NAMES, 4)
    # test__update_parent_reaction_data(MECH_BUTENE, MECH_BUTENE_SUBSET, 6, 9)
    # test__display(MECH_EMPTY, None, None)
    test__network(MECH_BUTENE_WITH_EXCLUDED_REACTIONS)
