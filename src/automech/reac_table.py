"""Functions acting on reactions dataframes."""

from collections.abc import Mapping, Sequence

import polars

from . import data
from .schema import Reaction
from .util import df_


def with_reaction_key(
    rxn_df: polars.DataFrame,
    col_name: str = "key",
    spc_key_dct: dict[str, object] | None = None,
) -> polars.DataFrame:
    """Add a key for identifying unique reactions to this dataframe.

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

    def _key(rcts, prds):
        if spc_key_dct is not None:
            rcts = list(map(spc_key_dct.get, rcts))
            prds = list(map(spc_key_dct.get, prds))
        rcts, prds = sorted([sorted(rcts), sorted(prds)])
        return data.reac.write_chemkin_equation(rcts, prds)

    return df_.map_(rxn_df, (Reaction.reactants, Reaction.products), col_name, _key)


def translate_reagents(
    rxn_df: polars.DataFrame,
    trans: Sequence[object] | Mapping[object, object],
    trans_into: Sequence[object] | None = None,
    rcol_out: str = Reaction.reactants,
    pcol_out: str = Reaction.products,
) -> polars.DataFrame:
    """Translate the reagent names in a reactions dataframe.

    :param rxn_df: A reactions dataframe
    :param trans: A translation mapping or a sequence of values to replace
    :param trans_into: If `trans` is a sequence, a sequence of values to replace by,
        defaults to None
    :param rcol_out: The column name to use for the reactants
    :param pcol_out: The column name to use for the products
    :return: The updated reactions dataframe
    """

    def _translate(col_in: str, col_out: str) -> polars.Expr:
        return (
            polars.col(col_in)
            .list.eval(polars.element().replace(old=trans, new=trans_into))
            .alias(col_out)
        )

    return rxn_df.with_columns(
        _translate(Reaction.reactants, rcol_out),
        _translate(Reaction.products, pcol_out),
    )
