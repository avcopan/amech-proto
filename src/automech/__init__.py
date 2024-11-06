"""Data processing at the level of whole mechanisms."""

from . import data, io, schema, util
from ._mech import (
    Mechanism,
    add_reactions,
    are_equivalent,
    display,
    display_reactions,
    display_species,
    expand_parent_stereo,
    expand_stereo,
    from_data,
    from_smiles,
    from_string,
    rate_units,
    reaction_count,
    reaction_equations,
    reactions,
    remove_all_reactions,
    rename,
    rename_dict,
    set_rate_units,
    set_reactions,
    set_species,
    set_thermo_temperatures,
    species,
    species_count,
    species_names,
    string,
    thermo_temperatures,
    update_parent_rates,
    update_parent_thermo,
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
    "thermo_temperatures",
    "rate_units",
    # setters
    "set_species",
    "set_reactions",
    "set_thermo_temperatures",
    "set_rate_units",
    # properties
    "species_count",
    "reaction_count",
    "species_names",
    "reaction_equations",
    "rename_dict",
    # transformations
    "rename",
    "remove_all_reactions",
    "add_reactions",
    "with_species",
    "without_unused_species",
    "expand_stereo",
    "expand_parent_stereo",
    "update_parent_thermo",
    "update_parent_rates",
    # comparisons
    "are_equivalent",
    # read/write,
    "string",
    "from_string",
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
