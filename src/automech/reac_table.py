"""Functions acting on reactions DataFrames."""

from collections.abc import Mapping, Sequence

import automol
import more_itertools as mit
import polars

from . import data, schema
from .schema import Reaction, ReactionMisc, ReactionRate
from .util import df_

DEFAULT_REAGENT_SEPARATOR = " + "


# update
def update_rates(
    rxn_df: polars.DataFrame, src_rxn_df: polars.DataFrame
) -> polars.DataFrame:
    """Read thermochemical data from one dataframe into another.

    :param rxn_df: Reactions DataFrame
    :param src_rxn_df: Reactions DataFrame with thermochemical data
    :return: reactions DataFrame
    """
    rxn_df = rxn_df.rename({ReactionRate.rate: ReactionMisc.orig_rate}, strict=False)

    if has_colliders(rxn_df):
        raise NotImplementedError(
            f"Updating rates with colliders not yet implemented.\n{rxn_df}"
        )

    col_key = df_.temp_column()
    rxn_df = with_reaction_key(rxn_df, col_name=col_key)
    src_rxn_df = with_reaction_key(src_rxn_df, col_name=col_key)
    rxn_df = rxn_df.join(src_rxn_df, how="left", on=col_key)
    rxn_df = rxn_df.drop(col_key)
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
    rcts = rxn_df[Reaction.reactants].to_list()
    prds = rxn_df[Reaction.products].to_list()
    return sorted(mit.unique_everseen(rcts + prds))


def reagent_strings(
    rxn_df: polars.DataFrame, separator: str = DEFAULT_REAGENT_SEPARATOR
) -> list[str]:
    """Get reagents as strings.

    :param rxn_df: A reactions DataFrame
    :param separator: The separator for joining reagent strings
    :return: The reagents as strings
    """
    return [separator.join(r) for r in reagents(rxn_df)]


# transformations
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
    rxn_df: polars.DataFrame, col_name: str, species_names: Sequence[str]
) -> polars.DataFrame:
    """Add a column indicating the presence of one or more species.

    :param rxn_df: A reactions DataFrame
    :param species_names: Species names
    :param col_name: The column name
    :return: The modified reactions DataFrame
    """
    return rxn_df.with_columns(
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(species_names))
        .list.any()
        .alias(col_name)
    )


def with_reagent_strings_column(
    rxn_df: polars.DataFrame, col_name: str, separator: str = DEFAULT_REAGENT_SEPARATOR
) -> polars.DataFrame:
    """Add a column containing the reagent strings on either side of the reaction.

    e.g. ["C2H6 + OH", "C2H5 + H2O"]

    :param rxn_df: A reactions DataFrame
    :param col_name: The column name
    :param separator: The separator for joining reagent strings
    :return: The reactions DataFrame with this extra column
    """
    return rxn_df.with_columns(
        polars.concat_list(
            polars.col(Reaction.reactants).list.join(separator),
            polars.col(Reaction.products).list.join(separator),
        ).alias(col_name)
    )


def with_reaction_key(
    rxn_df: polars.DataFrame,
    col_name: str = "key",
    spc_key_dct: dict[str, object] | None = None,
) -> polars.DataFrame:
    """Add a key for identifying unique reactions to this DataFrame.

    The key is formed by sorting reactants and products and then sorting the direction
    of the reaction.

        id = hash(sorted([sorted(rcts), sorted(prds)]))

    By default, this uses the species names, but a dictionary can be passed in to
    translate these into other species identifiers.

    :param rxn_df: A reactions DataFrame
    :param col_name: The column name
    :param spc_key_dct: A dictionary mapping species names onto unique species keys
    :return: A reactions DataFrame with this key as a new column
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
    """Translate the reagent names in a reactions DataFrame.

    :param rxn_df: A reactions DataFrame
    :param trans: A translation mapping or a sequence of values to replace
    :param trans_into: If `trans` is a sequence, a sequence of values to replace by,
        defaults to None
    :param rcol_out: The column name to use for the reactants
    :param pcol_out: The column name to use for the products
    :return: The updated reactions DataFrame
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
