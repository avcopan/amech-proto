"""Test automech functions."""
import automech


def test__from_smiles():
    """Test automech.from_smiles."""
    # Example 1
    mech = automech.from_smiles(
        ["CCC", "[OH]", "CC[CH2]", "O"],
        rxn_smis=["CCC.[OH]>>CC[CH2].O"],
        name_dct={"CCC": "C3H8", "CC[CH2]": "C3H7y1"},
    )
    print(mech)

    # Example 2
    mech = automech.from_smiles(["CCC"], rxn_smis=[])
    print(mech)

    # Example 3 (empty mechanism)
    mech = automech.from_smiles([], rxn_smis=[])
    print(mech)
