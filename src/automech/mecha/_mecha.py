"""Primary mechanism processing routines
"""

import itertools
import os
from typing import Optional, Tuple, Union

import automol
import pandas
from tqdm.auto import tqdm

from mecha import data, schema
from mecha.io import chemkin
from mecha.schema import Reactions, Species
from mecha.util import df_

tqdm.pandas()


def display(
    inp: Union[pandas.DataFrame, str],
    spc_inp: Union[pandas.DataFrame, str],
    stereo: bool = True,
    exclude: tuple = (
        {"H": None},
        {"O": 1, "H": None},
        {"O": 2, "H": None},
        {"C": 1, "H": None},
    ),
    out: str = "example.html",
):
    """Display a mechanism as a reaction network

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param keys: Keys of extra columns to print
    :param stereo: Display with stereochemistry?
    :param exclude: A list of formulas to exclude; Use `None` for wildcard value
    :param out: The HTML file to write to
    """
    from IPython import display as ipd
    from pyvis.network import Network

    rxn_df = df_.from_csv(inp) if isinstance(inp, str) else inp
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp

    rxn_df = schema.validate_reactions(rxn_df)
    spc_df = schema.validate_species(spc_df, smi=True)

    # Handle excluded species
    def _is_excluded(chi: str):
        fml = automol.amchi.formula(chi)
        if any(automol.form.match(fml, f) for f in exclude):
            return True
        return False

    spc_df["excluded"] = spc_df[Species.chi].progress_apply(_is_excluded)
    excl_names = list(spc_df[spc_df["excluded"]][Species.name])

    image_dir = "images"
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    def _create_image(chi):
        gra = automol.amchi.graph(chi, stereo=stereo)
        chk = automol.amchi.amchi_key(chi)
        svg_str = automol.graph.svg_string(gra, image_size=100)

        path = os.path.join(image_dir, f"{chk}.svg")
        with open(path, mode="w") as file:
            file.write(svg_str)

        return path

    spc_df["image_path"] = spc_df[Species.chi].progress_apply(_create_image)

    net = Network(directed=True, notebook=True)

    def _add_node(row: pandas.Series):
        name = row[Species.name]
        smi = row[Species.smi]
        path = row["image_path"]
        if name not in excl_names:
            net.add_node(name, shape="image", image=path, title=smi)

    def _add_edge(row: pandas.Series):
        eq = row[Reactions.eq]
        rnames, pnames, _ = data.reac.read_chemkin_equation(eq)
        for rname, pname in itertools.product(rnames, pnames):
            if rname not in excl_names and pname not in excl_names:
                net.add_edge(rname, pname, title=eq)

    spc_df.progress_apply(_add_node, axis=1)
    rxn_df.progress_apply(_add_edge, axis=1)
    ipd.display(net.show(out))


def display_reactions(
    inp: Union[pandas.DataFrame, str],
    spc_inp: Union[pandas.DataFrame, str],
    keys: Tuple[str, ...] = (Reactions.eq,),
    stereo: bool = True,
    eqs: Optional[list] = None,
    spc_keys: Tuple[str, ...] = (Species.smi,),
):
    """Display the reactions in a mechanism

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param keys: Keys of extra columns to print
    :param stereo: Display with stereochemistry?
    :param eqs: Optionally, specify specific equations to visualize
    :param spc_keys: Optionally, translate the reactant and product names into these
        species dataframe values
    """
    rxn_df = df_.from_csv(inp) if isinstance(inp, str) else inp
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp

    rxn_df = schema.validate_reactions(rxn_df)
    spc_df = schema.validate_species(spc_df)

    chi_dct = df_.lookup_dict(spc_df, Species.name, Species.chi)
    trans_dcts = {k: df_.lookup_dict(spc_df, Species.name, k) for k in spc_keys}

    def display_(row):
        # Print the requested information
        for key in keys:
            val = row[key] if key in row else "No such column"
            print(f"{key}: {val}")

        # Display the reaction
        eq = row[Reactions.eq]
        rchis, pchis, _ = data.reac.read_chemkin_equation(eq, trans_dct=chi_dct)

        for key, trans_dct in trans_dcts.items():
            rvals, pvals, _ = data.reac.read_chemkin_equation(eq, trans_dct=trans_dct)
            print(f"Species `name`=>`{key}` translation")
            print(f"  reactants = {rvals}")
            print(f"  products = {pvals}")

        if not all(isinstance(n, str) for n in rchis + pchis):
            print(f"Some ChIs missing from species table: {rchis} = {pchis}")
        else:
            automol.amchi.display_reaction(rchis, pchis, stereo=stereo)

    if eqs is not None:
        eqs = list(map(data.reac.standardize_chemkin_equation, eqs))
        rxn_df = rxn_df[rxn_df[Reactions.eq].isin(eqs)]

    rxn_df.apply(display_, axis=1)


def classify_reactions(
    inp: Union[pandas.DataFrame, str],
    spc_inp: Union[pandas.DataFrame, str],
    out: Optional[str] = None,
    err_out: Optional[str] = None,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """Classify the reactions in a mechanism

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A species table, as a CSV file path or dataframe
    :param out: Optionally, write the reaction data output to this file path
    :param err_out: Optionally, write the error data output to this file path
    :return: A dataframe of classified reactions, and a dataframe of error cases
    """
    rxn_df = df_.from_csv(inp) if isinstance(inp, str) else inp
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp

    rxn_df = schema.validate_reactions(rxn_df)
    spc_df = schema.validate_species(spc_df)

    rxn_df = combine_duplicate_reactions(rxn_df, first=False)

    # Do the classification with a progress bar, since it may take a while
    chi_dct = df_.lookup_dict(spc_df, Species.name, Species.chi)

    def objs_(row):
        eq = row[Reactions.eq]
        rchis, pchis, _ = data.reac.read_chemkin_equation(eq, trans_dct=chi_dct)
        objs = automol.reac.from_amchis(rchis, pchis, stereo=False)
        return objs if objs else pandas.NA

    rxn_df[Reactions.obj] = rxn_df.progress_apply(objs_, axis=1)

    # Separate out the unclassified reactions
    err_df = rxn_df[rxn_df[Reactions.obj].isna()].drop(columns=[Reactions.obj])
    rxn_df = rxn_df[rxn_df[Reactions.obj].notna()].drop(columns=[Reactions.rate])

    # Expand reactions with multiple possible mechanisms
    def amchi_(row):
        obj = row[Reactions.obj]
        return automol.reac.ts_amchi(obj)

    rxn_df = expand_duplicate_reactions(rxn_df)
    rxn_df[Reactions.chi] = rxn_df.progress_apply(amchi_, axis=1)

    # Expand duplicates among the unclassified reactions again
    err_df = expand_duplicate_reactions(err_df)

    rxn_df = schema.validate_reactions(rxn_df)
    err_df = schema.validate_reactions(err_df)
    df_.to_csv(rxn_df, out)
    df_.to_csv(err_df, err_out)

    return rxn_df, err_df


def expand_species_stereo(
    inp: Union[pandas.DataFrame, str],
    out: Optional[str] = None,
    enant: bool = True,
) -> pandas.DataFrame:
    """Stereoexpand a list of species

    :param inp: A species table, as a CSV file path or dataframe
    :param out: Optionally, write the species data output to this file path
    :param enant: Distinguish between enantiomers?, defaults to True
    :return: The stereo-expanded species dataframe
    """
    spc_df = df_.from_csv(inp) if isinstance(inp, str) else inp

    def expand_amchi_(chi):
        return automol.amchi.expand_stereo(chi, enant=enant)

    def name_(row):
        name = row[Species.orig_name]
        chi = row[Species.chi]
        return data.name.with_stereo_suffix(name, chi, racem=not enant)

    spc_df = schema.validate_species(spc_df)
    spc_df = spc_df.rename(columns=dict(zip(schema.S_CURR_COLS, schema.S_ORIG_COLS)))
    spc_df[Species.chi] = spc_df[Species.orig_chi].progress_apply(expand_amchi_)
    spc_df = spc_df.explode(Species.chi)
    spc_df[Species.name] = spc_df.apply(name_, axis=1)

    spc_df = schema.validate_species(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df


def expand_reaction_stereo(
    inp: Union[pandas.DataFrame, str],
    spc_inp: Union[pandas.DataFrame, str],
    out: Optional[str] = None,
    err_out: Optional[str] = None,
    enant: bool = True,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """Stereoexpand a list of reactions

    Requires that the reactions have been classified and the species have been expanded

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A dataframe or CSV filepath with stereoexpanded species data
    :param out: Optionally, write the reaction data output to this file path
    :param err_out: Optionally, write the error data output to this file path
    :param enant: Distinguish between enantiomers?, defaults to True
    :return: A dataframe of stereoexpanded reactions, and a dataframe of error cases
    """
    rxn_df = df_.from_csv(inp) if isinstance(inp, str) else inp
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp

    if not enant:
        raise NotImplementedError("Reduced expansion not yet implemented")

    rxn_df = schema.validate_reactions(rxn_df)
    rxn_df = rxn_df.rename(columns=dict(zip(schema.R_CURR_COLS, schema.R_ORIG_COLS)))

    # Expand reaction objects
    def objs_(obj):
        if isinstance(obj, str):
            obj = automol.reac.from_string(obj)
        return automol.reac.expand_stereo(obj, enant=enant)

    rxn_df[Reactions.obj] = rxn_df[Reactions.obj].progress_apply(objs_)

    # Get stereo-resolved reaction equations (initially as lists)
    name_dct = df_.lookup_dict(spc_df, [Species.orig_name, Species.chi], Species.name)

    def eq_(row):
        eq0 = row[Reactions.orig_eq]
        objs = row[Reactions.obj]
        eqs = []
        for obj in objs:
            rname0s, pname0s, _ = data.reac.read_chemkin_equation(eq0)
            rchis, pchis = automol.reac.amchis(obj)
            rnames = tuple(map(name_dct.get, zip(rname0s, rchis)))
            pnames = tuple(map(name_dct.get, zip(pname0s, pchis)))
            if not all(isinstance(n, str) for n in rnames + pnames):
                return pandas.NA
            eqs.append(data.reac.write_chemkin_equation(rnames, pnames))
        return tuple(eqs)

    rxn_df[Reactions.eq] = rxn_df.progress_apply(eq_, axis=1)

    # Separate out the reactions that had errors
    err_df = rxn_df[rxn_df[Reactions.eq].isna()]
    rxn_df = rxn_df[rxn_df[Reactions.eq].notna()]

    err_df = err_df.drop(columns=[Reactions.eq]).rename(
        columns={Reactions.orig_eq: Reactions.eq}
    )
    rxn_df = rxn_df.explode([Reactions.eq, Reactions.obj])

    rxn_df = schema.validate_reactions(rxn_df)
    err_df = schema.validate_reactions(err_df)
    df_.to_csv(rxn_df, out)
    df_.to_csv(err_df, err_out)

    return rxn_df, err_df


def expand_stereo(
    inp: Union[pandas.DataFrame, str],
    spc_inp: Union[pandas.DataFrame, str],
    out: Optional[str] = None,
    err_out: Optional[str] = None,
    spc_out: Optional[str] = None,
    enant: bool = True,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """Stereoexpand a mechanism (both species and reactions)

    :param inp: A reactions table, as a CSV file path or dataframe
    :param spc_inp: A dataframe or CSV filepath with stereoexpanded species data
    :param out: Optionally, write the reaction data output to this file path
    :param err_out: Optionally, write the error data output to this file path
    :param enant: Distinguish between enantiomers?, defaults to True
    :return: Dataframes of stereoexpanded reactions and species, and a dataframe of
        error cases for the reactions
    """
    spc_df = expand_species_stereo(spc_inp, out=spc_out, enant=enant)
    rxn_df, err_df = expand_reaction_stereo(
        inp, spc_inp, out=out, err_out=err_out, enant=enant
    )
    return rxn_df, spc_df, err_df


# Conversions to/from mechanalyzer formats
def from_mechanalyzer(
    inp: str,
    spc_inp: Union[pandas.DataFrame, str],
    out: Optional[str] = None,
    spc_out: Optional[str] = None,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """Import from mechanalyzer format

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param spc_inp: A mechanalyzer-style species table, as a dataframe or CSV file path
    :param out: Optionally, write the reaction data output to this file path
    :param spc_out: Optionally, write the species data output to this file path
    :return: Dataframes for the reactions and species
    """
    rxn_df = chemkin.read.reactions(inp)
    spc_df = df_.from_csv(spc_inp) if isinstance(spc_inp, str) else spc_inp
    spc_df = spc_df.rename(columns={"inchi": Species.chi, "smiles": Species.smi})

    rxn_df = schema.validate_reactions(rxn_df)
    spc_df = schema.validate_species(spc_df)

    df_.to_csv(rxn_df, out)
    df_.to_csv(spc_df, spc_out)

    return rxn_df, spc_df


def to_mechanalyzer(
    rxn_df: pandas.DataFrame, spc_df: pandas.DataFrame, drop: bool = False
) -> Tuple[str, str]:
    """Get old reaction and species dictionaries for a mechanism

    :param rxn_df: The reactions dataframe
    :param spc_df: The species dataframe
    :param drop: Drop species which have no reactions?, defaults to False
    :return: The mechanism and species list, in CHEMKIN and CSV formats, respectively
    """
    import chemkin_io

    from mechanalyzer.parser import spc as ma_species_io_

    rate0 = data.rate.SimpleRate()
    headers0 = [
        "name",
        "smiles",
        "inchi",
        "inchikey",
        "mult",
        "charge",
        "canon_enant_ich",
    ]

    rxn_df = schema.validate_reactions(rxn_df)
    spc_df = schema.validate_species(spc_df)

    # 1. Form the reactions dictionary
    def reaction_dict_item_(row):
        eq = row[Reactions.eq]
        rate = row[Reactions.rate] if Reactions.rate in row else rate0
        rxn = data.reac.from_equation(eq, rate)
        key = data.reac.chemkin_reagents(rxn, tuple_coll=True)
        val = data.rate.to_old_object(data.reac.rate(rxn))
        return (key, val)

    rxn_dct = dict(rxn_df.progress_apply(reaction_dict_item_, axis=1).to_list())

    # 2. Drop unused species, if requested
    if drop:
        spc_set = {s for r, p, _ in rxn_dct for s in r + p}
        spc_df = spc_df[spc_df[Species.name].isin(spc_set)]
        assert (
            set(spc_df[Species.name]) == spc_set
        ), f'Missing species: {set(spc_df["name"]) - spc_set}'

    # 3. Re-format the species dataframe
    spc_df = schema.validate_species(spc_df, smi=True)
    spc_df[Species.chi] = spc_df[Species.chi].progress_apply(automol.chi.inchi_to_amchi)
    spc_df = spc_df.rename(columns={Species.chi: "inchi", Species.smi: "smiles"})

    # 4. Make sure we have all of the usual columns
    spc_df["fml"] = spc_df["inchi"].progress_apply(automol.amchi.formula)
    spc_df["inchikey"] = spc_df["inchi"].progress_apply(automol.chi.inchi_key)
    spc_df["canon_enant_ich"] = spc_df["inchi"].progress_apply(
        automol.chi.canonical_enantiomer
    )
    spc_df = spc_df.drop(columns=spc_df.columns.difference(headers0))

    # 5. Add in the basis species
    basis_dct = {
        "H2": automol.smiles.inchi("[H][H]"),
        "H2O": automol.smiles.inchi("O"),
        "CH4": automol.smiles.inchi("C"),
    }
    for name, chi in basis_dct.items():
        if chi not in spc_df["inchi"].values:
            row_dct = {
                "name": name,
                "smiles": automol.amchi.smiles(chi),
                "inchi": chi,
                "inchikey": automol.chi.inchi_key(chi),
                "mult": 1,
                "charge": 0,
                "canon_enant_ich": automol.chi.canonical_enantiomer(chi),
            }
            row = pandas.DataFrame([row_dct])
            spc_df = pandas.concat([row, spc_df], ignore_index=True)

    # 6. Move the basis species to the top
    in_basis = spc_df["inchi"].isin(basis_dct.values())
    spc_df = pandas.concat([spc_df[in_basis], spc_df[~in_basis]], ignore_index=True)

    # 7. Form the species dictionary
    spc_df = spc_df[~spc_df["name"].duplicated(keep="first")]
    spc_df = spc_df.set_index("name")
    spc_dct = spc_df.to_dict("index")

    # 8. Write the mechanism string
    mech_str = chemkin_io.writer.mechanism.write_chemkin_file(
        mech_spc_dct=spc_dct, rxn_param_dct=rxn_dct
    )

    # 9. Write the species string
    spc_str = ma_species_io_.csv_string(spc_df.to_dict("index"), spc_df.columns)
    return mech_str, spc_str


# Helpers
def combine_duplicate_reactions(
    rxn_df: pandas.DataFrame, first: bool = False
) -> pandas.DataFrame:
    """Combine duplicate reactions, so each reaction only appears once

    :param rxn_df: The reactions dataframe
    :param first: Keep only the first column value? Otherwise, the values will be
        grouped into lists
    :return: The reactions dataframe
    """
    agg_ = "first" if first else list
    agg_dct = {c: agg_ for c in schema.DUP_DIFF_COLS if c in rxn_df}
    agg_dct.update(
        {c: "first" for c in rxn_df.columns if c not in schema.DUP_DIFF_COLS}
    )
    return rxn_df.groupby(Reactions.eq, as_index=False).agg(agg_dct)


def expand_duplicate_reactions(rxn_df: pandas.DataFrame) -> pandas.DataFrame:
    """Expand duplicate reactions, so they appear multiple times

    :param rxn_df: The reactions dataframe
    :return: The reactions dataframe
    """
    exp_cols = [c for c in schema.DUP_DIFF_COLS if c in rxn_df]
    rxn_df[exp_cols] = rxn_df[exp_cols].map(list)
    return rxn_df.explode(exp_cols)
