"""CHEMKIN name handling.

CHEMKIN naming scheme idea:
- Break the 16 characters available into blocks...
    * formula, connectivity, bond stereo, atom stereo, mirroring
    * as part of formula, describe functional groups using one-letter codes
        # functional group atoms would be removed from the formula
        # specify double bonds and missing hydrogens as functional groups:
            @ Y2 = 2-yl
            @ E3 = 3-ene
        # example: C5Y2E3P4 = pentane 2-yl 3-ene 4-peroxy
        # we would probably need to use InChI/AMChI canonical numbers for these
    * use alternating upper and lower case values, to save space on dashes
        # example: C5Y2E3P4upXYetc
    * After a max number of characters, replace the block value with a hash
    * When only one connectivity is possible, omit the connectivity hash
        # example: H2O
"""

import automol


def with_stereo_suffix(name: str, chi: str, racem: bool = False) -> str:
    """Append a stereo suffix to a CHEMKIN name, based on its ChI string.

    :param name: The CHEMKIN name
    :param chi: The ChI string
    :param racem: Use a racemic suffix?, defaults to False
    :return: The CHEMKIN name, with stereo suffix
    """
    bpar_dct = automol.amchi.bond_stereo_parities(chi, ordered_key=True)
    apar_dct = automol.amchi.atom_stereo_parities(chi)
    is_inv = automol.amchi.is_inverted_enantiomer(chi)
    is_enant = automol.amchi.is_enantiomer(chi)
    bpars = list(map(bpar_dct.get, sorted(bpar_dct)))
    apars = list(map(apar_dct.get, sorted(apar_dct)))
    bsuff = "".join("e" if p else "z" for p in bpars)
    asuff = "".join("s" if p else "r" for p in apars)
    isuff = ""
    if is_enant:
        isuff = "R" if racem else ("1" if is_inv else "0")
    return "".join((name, bsuff, asuff, isuff))
