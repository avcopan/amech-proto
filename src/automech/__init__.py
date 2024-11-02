"""Data processing at the level of whole mechanisms."""

from . import data, io, schema, util
from ._mech import (
    Mechanism,
    are_equivalent,
    display,
    display_reactions,
    display_species,
    expand_parent_stereo,
    expand_stereo,
    from_data,
    from_smiles,
    reacting_species_names,
    reaction_count,
    reactions,
    rename,
    rename_dict,
    set_reactions,
    set_species,
    species,
    species_count,
    with_species,
    without_unused_species,
)

__all__ = [
    # types
    "Mechanism",
    # functions
    "from_data",
    "from_smiles",
    # getters
    "species",
    "reactions",
    # setters
    "set_species",
    "set_reactions",
    # properties
    "species_count",
    "reaction_count",
    "reacting_species_names",
    "rename_dict",
    # transformations
    "rename",
    "with_species",
    "without_unused_species",
    "expand_stereo",
    "expand_parent_stereo",
    # comparisons
    "are_equivalent",
    # display
    "display",
    "display_species",
    "display_reactions",
    # modules
    "io",
    "data",
    "util",
    "schema",
]
