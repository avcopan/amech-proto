"""Functions acting on reactions dataframes."""

import polars

from . import data
from .schema import Reaction
from .util import df_


def with_reaction_key(
    rxn_df: polars.DataFrame,
    col_name: str = "key",
    spc_key_dct: dict[str, object] | None = None,
) -> polars.DataFrame:
    """Addd a key for identifying unique reactions to this dataframe.

    The key is formed by sorting reactants and products and then sorting the direction
    of the reaction.

        id = hash(sorted([sorted(rcts), sorted(prds)]))

    By default, this uses the species names, but a dictionary can be passed in to
    translate these into other species identifiers.

    :param rxn_df: A reactions dataframe
    :param col_name: The column name for the key, defaults to "key"
    :param spc_key_dct: A dictionary mapping species names onto unique species keys
    :return: A reactions dataframe with this key as a new column
    """

    def _key(eq):
        rcts, prds, *_ = data.reac.read_chemkin_equation(eq, trans_dct=spc_key_dct)
        rcts, prds = sorted([sorted(rcts), sorted(prds)])
        return data.reac.write_chemkin_equation(rcts, prds)

    return df_.map_(rxn_df, Reaction.eq, col_name, _key)
