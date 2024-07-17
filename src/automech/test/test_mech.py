"""Test automech functions."""

import automech


def test__from_smiles():
    """Test automech.from_smiles.

    Also tests the display function, to make sure it runs.
    """
    # Example 1
    mech = automech.from_smiles(
        ["CCC", "[OH]", "CC[CH2]", "O"],
        rxn_smis=["CCC.[OH]>>CC[CH2].O"],
        name_dct={"CCC": "C3H8", "CC[CH2]": "C3H7y1"},
    )
    print(mech)
    automech.display(mech, open_browser=False)

    # Example 2
    mech = automech.from_smiles(["CCC"], rxn_smis=[])
    print(mech)
    automech.display(mech, open_browser=False)

    # Example 3 (empty mechanism)
    mech = automech.from_smiles([], rxn_smis=[])
    print(mech)
    automech.display(mech, open_browser=False)


if __name__ == "__main__":
    test__from_smiles()
