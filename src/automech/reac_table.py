"""Functions acting on reactions DataFrames."""

import itertools
from collections.abc import Mapping, Sequence

import automol
import more_itertools as mit
import polars

from . import data, schema
from .schema import Reaction, ReactionRate
from .util import col_, df_

m_col_ = col_

DEFAULT_REAGENT_SEPARATOR = " + "


# update
def left_update(
    rxn_df: polars.DataFrame,
    src_rxn_df: polars.DataFrame,
    drop_orig: bool = False,
) -> polars.DataFrame:
    """Left-update reaction data by reaction key.

    :param rxn_df: reaction DataFrame
    :param src_rxn_df: Source reaction DataFrame
    :param drop_orig: Whether to drop original column values
    :return: Reaction DataFrame
    """
    drop_cols = m_col_.orig(schema.columns(Reaction))

    # Add reaction keys
    tmp_col = df_.temp_column()
    rxn_df = with_reaction_key(rxn_df, tmp_col)
    src_rxn_df = with_reaction_key(src_rxn_df, tmp_col)

    # Update
    rxn_df = df_.left_update(rxn_df, src_rxn_df, col_=tmp_col, drop_orig=drop_orig)

    # Drop unnecessary columns
    rxn_df = rxn_df.drop(tmp_col, *drop_cols, strict=False)
    return rxn_df


def left_update_rates(
    rxn_df: polars.DataFrame, src_rxn_df: polars.DataFrame
) -> polars.DataFrame:
    """Read thermochemical data from one dataframe into another.

    (AVC note: I think this can be deprecated and replaced with the more general
    function above...)

    :param rxn_df: Reactions DataFrame
    :param src_rxn_df: Reactions DataFrame with thermochemical data
    :return: reactions DataFrame
    """
    rxn_df = rxn_df.rename(col_.to_orig(ReactionRate.rate), strict=False)

    if has_colliders(rxn_df):
        raise NotImplementedError(
            f"Updating rates with colliders not yet implemented.\n{rxn_df}"
        )

    col_key = df_.temp_column()
    rxn_df = with_reaction_key(rxn_df, col=col_key)
    src_rxn_df = with_reaction_key(src_rxn_df, col=col_key)
    rxn_df = rxn_df.join(src_rxn_df, how="left", on=col_key)
    rxn_df = rxn_df.drop(col_key, polars.selectors.ends_with("_right"))
    rxn_df, *_ = schema.reaction_table(rxn_df, model_=ReactionRate)
    return rxn_df


# properties
def has_colliders(rxn_df: polars.DataFrame) -> bool:
    """Determine whether a reactions DataFrame has colliders.

    :param rxn_df: Reactions DataFrame
    :return: `True` if it does, `False` if not
    """
    return ReactionRate.colliders in rxn_df and df_.has_values(
        rxn_df.get_column(ReactionRate.colliders).struct.unnest()
    )


def reagents(rxn_df: polars.DataFrame) -> list[list[str]]:
    """Get reagents as lists.

    :param rxn_df: A reactions DataFrame
    :return: The reagents
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(rcts + prds))


def species(rxn_df: polars.DataFrame) -> list[str]:
    """Get species in reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Species names
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(itertools.chain.from_iterable(rcts + prds)))


def reagent_strings(
    rxn_df: polars.DataFrame, sep: str = DEFAULT_REAGENT_SEPARATOR
) -> list[str]:
    """Get reagents as strings.

    :param rxn_df: A reactions DataFrame
    :param sep: The separator for joining reagent strings
    :return: The reagents as strings
    """
    return [sep.join(r) for r in reagents(rxn_df)]


# add columns
def with_reaction_key(
    rxn_df: polars.DataFrame,
    col: str = "key",
    spc_key_dct: dict[str, object] | None = None,
) -> polars.DataFrame:
    """Add a key for identifying unique reactions to this DataFrame.

    The key is formed by sorting reactants and products and then sorting the direction
    of the reaction.

        id = string join(sorted([sorted(rcts), sorted(prds)]))

    By default, this uses the species names, but a dictionary can be passed in to
    translate these into other species identifiers.

    :param rxn_df: A reactions DataFrame
    :param col: The column name
    :param spc_key_dct: A dictionary mapping species names onto unique species keys
    :return: A reactions DataFrame with this key as a new column
    """

    def _key(rcts, prds):
        if spc_key_dct is not None:
            rcts = list(map(spc_key_dct.get, rcts))
            prds = list(map(spc_key_dct.get, prds))
        rcts, prds = sorted([sorted(rcts), sorted(prds)])
        return data.reac.write_chemkin_equation(rcts, prds)

    return df_.map_(rxn_df, (Reaction.reactants, Reaction.products), col, _key)


def with_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Add placeholder rate data to this DataFrame, if missing.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    if ReactionRate.rate not in rxn_df:
        rate = dict(data.rate.SimpleRate())
        rxn_df = rxn_df.with_columns(polars.lit(rate).alias(ReactionRate.rate))

    if ReactionRate.colliders not in rxn_df:
        coll = {"M": None}
        rxn_df = rxn_df.with_columns(polars.lit(coll).alias(ReactionRate.colliders))

    return rxn_df


def without_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Remove rate data from this DataFrame, if present.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    if ReactionRate.rate not in rxn_df:
        rxn_df = rxn_df.drop(ReactionRate.rate)

    if ReactionRate.colliders not in rxn_df:
        rxn_df = rxn_df.drop(ReactionRate.colliders)

    return rxn_df


def with_species_presence_column(
    rxn_df: polars.DataFrame, col: str, species_names: Sequence[str]
) -> polars.DataFrame:
    """Add a column indicating the presence of one or more species.

    :param rxn_df: A reactions DataFrame
    :param species_names: Species names
    :param col: The column name
    :return: The modified reactions DataFrame
    """
    return rxn_df.with_columns(
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(species_names))
        .list.any()
        .alias(col)
    )


def with_reagent_strings_column(
    rxn_df: polars.DataFrame, col: str, sep: str = DEFAULT_REAGENT_SEPARATOR
) -> polars.DataFrame:
    """Add a column containing the reagent strings on either side of the reaction.

    e.g. ["C2H6 + OH", "C2H5 + H2O"]

    :param rxn_df: A reactions DataFrame
    :param col: The column name
    :param sep: The separator for joining reagent strings
    :return: The reactions DataFrame with this extra column
    """
    return rxn_df.with_columns(
        polars.concat_list(
            polars.col(Reaction.reactants).list.join(sep),
            polars.col(Reaction.products).list.join(sep),
        ).alias(col)
    )


def rename(
    rxn_df: polars.DataFrame,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    drop_orig: bool = False,
) -> polars.DataFrame:
    """Rename species in a reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :return: Reactions DataFrame
    """
    col_dct = col_.to_orig([Reaction.reactants, Reaction.products])
    rxn_df = rxn_df.with_columns(polars.col(c0).alias(c) for c0, c in col_dct.items())
    rxn_df = translate_reagents(rxn_df=rxn_df, trans=names, trans_into=new_names)
    if drop_orig:
        rxn_df = rxn_df.drop(col_dct.values())
    return rxn_df


def translate_reagents(
    rxn_df: polars.DataFrame,
    trans: Sequence[object] | Mapping[object, object],
    trans_into: Sequence[object] | None = None,
    rct_col: str = Reaction.reactants,
    prd_col: str = Reaction.products,
) -> polars.DataFrame:
    """Translate the reagent names in a reactions DataFrame.

    :param rxn_df: A reactions DataFrame
    :param trans: A translation mapping or a sequence of values to replace
    :param trans_into: If `trans` is a sequence, a sequence of values to replace by,
        defaults to None
    :param rct_col: The column name to use for the reactants
    :param prd_col: The column name to use for the products
    :return: The updated reactions DataFrame
    """
    expr = (
        polars.element().replace(trans)
        if trans_into is None
        else polars.element().replace(trans, trans_into)
    )

    return rxn_df.with_columns(
        polars.col(Reaction.reactants).list.eval(expr).alias(rct_col),
        polars.col(Reaction.products).list.eval(expr).alias(prd_col),
    )


def select_pes(
    rxn_df: polars.DataFrame,
    formula_: str | dict | Sequence[str | dict],
    exclude: bool = False,
) -> polars.DataFrame:
    """Select (or exclude) PES by formula(s).

    :param rxn_df: Reaction DataFrame
    :param formula_: PES formula(s) to include or exclude
    :param exclude: Whether to exclude or include the formula(s)
    :return: Reaction DataFrame
    """
    formula_ = [formula_] if isinstance(formula_, str | dict) else formula_
    fmls = [automol.form.from_string(f) if isinstance(f, str) else f for f in formula_]

    def _match(fml: dict[str, int]) -> bool:
        return any(automol.form.match(fml, f) for f in fmls)

    col_tmp = df_.temp_column()
    rxn_df = df_.map_(rxn_df, Reaction.formula, col_tmp, _match)
    match_expr = polars.col(col_tmp)
    rxn_df = rxn_df.filter(~match_expr if exclude else match_expr)
    rxn_df = rxn_df.drop(col_tmp)
    return rxn_df
