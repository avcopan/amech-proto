"""Data processing at the level of whole mechanisms."""

from . import data, io, net, schema, util
from ._mech import (
    Mechanism,
    add_reactions,
    are_equivalent,
    difference,
    display,
    display_reactions,
    display_species,
    drop_duplicate_reactions,
    drop_self_reactions,
    enumerate_reactions,
    expand_parent_stereo,
    expand_stereo,
    from_data,
    from_network,
    from_smiles,
    from_string,
    intersection,
    neighborhood,
    network,
    rate_units,
    reaction_count,
    reaction_equations,
    reactions,
    remove_all_reactions,
    rename,
    rename_dict,
    select_pes,
    set_rate_units,
    set_reactions,
    set_species,
    set_thermo_temperatures,
    species,
    species_count,
    species_names,
    string,
    thermo_temperatures,
    update,
    with_intersection_columns,
    with_key,
    with_rates,
    with_sort_data,
    with_species,
    without_unused_species,
)

__all__ = [
    # types
    "Mechanism",
    # functions
    "from_data",
    "from_network",
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
    "network",
    # add/remove reactions
    "drop_duplicate_reactions",
    "drop_self_reactions",
    # transformations
    "rename",
    "remove_all_reactions",
    "add_reactions",
    "select_pes",
    "neighborhood",
    "with_species",
    "without_unused_species",
    "with_rates",
    "with_key",
    "expand_stereo",
    # binary operations
    "intersection",
    "difference",
    "update",
    "with_intersection_columns",
    # parent
    "expand_parent_stereo",
    # building
    "enumerate_reactions",
    # sorting
    "with_sort_data",
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
    "net",
]
