import os

import pandas as pd
import scipy.stats


def load_subject_rdms(sub_list, base_path, split="full"):
    """Load and combine subject RDM DataFrames.

    Parameters
    ----------
    sub_list : list of str
        Subject identifiers.
    base_path : str
        Directory containing subject RDM folders.
    split : {'full', 'block1', 'block2'}, default='full'
        Which RDM version to load.

    Returns
    -------
    pandas.DataFrame
        Combined RDMs indexed by ('sub', 'time') with columns named 'features'.
    """
    dfs = []

    for sub in sub_list:
        save_path = os.path.join(base_path, sub, "RDM", f"RDM_{sub}_{split}.pkl")

        # Load the subject RDM DataFrame.
        time_rdm_df = pd.read_pickle(save_path)

        # Round to prevent floating point errors.
        time_rdm_df.index = time_rdm_df.index.round(5)

        # Add 'sub' as a new index level.
        time_rdm_df["sub"] = sub
        time_rdm_df = time_rdm_df.set_index("sub", append=True)

        # Reorder index levels to ('sub', 'time').
        time_rdm_df = time_rdm_df.reorder_levels(["sub", "time"])

        dfs.append(time_rdm_df)

    # Concatenate along the index.
    combined_df = pd.concat(dfs)

    # Sort the index.
    combined_df.sort_index(inplace=True)

    # Set the column name to "features".
    combined_df.columns.name = "features"

    return combined_df


def load_model_rdms(model_conditions, model_dir, layer="layer1", series="separate"):
    """Load and combine model RDM DataFrames.

    Parameters
    ----------
    model_conditions : list of str
        List of model condition names (e.g., ["Untrained", "Supervised", ...]).
    model_dir : str
        Directory containing model RDM .pkl files.
    layer : {'layer1', 'layer2', 'layerall'}, default='layer1'
        Layer identifier included in the filename.
    series : {'separate', 'split'}, default='separate'
        Which RDM version to load.

    Returns
    -------
    pandas.DataFrame
        Combined RDMs indexed by ('model', 'time') with columns named 'features'.
    """
    dfs = []

    for condition in model_conditions:
        file_path = os.path.join(
            model_dir, f"{condition}_rdm_{series}_spearman_{layer}.pkl"
        )

        # Load the RDM DataFrame.
        rdm_df = pd.read_pickle(file_path)

        # Add 'model' as a new index level.
        rdm_df["model"] = condition
        rdm_df = rdm_df.set_index("model", append=True)

        # Reorder index levels to ('model', 'step').
        rdm_df = rdm_df.reorder_levels(["model", "step"])

        dfs.append(rdm_df)

    # Concatenate all RDMs.
    combined_df = pd.concat(dfs)

    # Sort the index and label the columns.
    combined_df.sort_index(inplace=True)
    combined_df.columns.name = "features"

    return combined_df


def compute_nested_correlation_df(df_a, df_b, corr_measure=scipy.stats.spearmanr):
    """Compute pairwise correlations between rows of two DataFrames.

    Parameters
    ----------
    df_a : pd.DataFrame
        First DataFrame with numeric columns. Can have a multi-level index.
    df_b : pd.DataFrame
        Second DataFrame with numeric columns. Can have a multi-level index.
    corr_measure : callable, optional
        Correlation function that takes two 1D arrays and returns a tuple
        (correlation, p-value). Default is `scipy.stats.spearmanr`.

    Returns
    -------
    pd.DataFrame
        Multi-indexed DataFrame containing correlations for all combinations
        of rows from `df_a` and `df_b`.
        The index levels are automatically taken from `df_a` and `df_b`.
        The column 'correlation' contains the correlation value between
        corresponding row vectors.
    """
    index_tuples = []
    correlations = []

    # Extract index names automatically.
    idx_a_names = df_a.index.names
    idx_b_names = df_b.index.names
    combined_idx_names = idx_a_names + idx_b_names

    # Iterate through rows.
    for idx_a, row_a in df_a.iterrows():
        vec_a = row_a.values

        for idx_b, row_b in df_b.iterrows():
            vec_b = row_b.values

            # Compute the correlation.
            corr = corr_measure(vec_a, vec_b)[0]

            # Ensure tuples even if using a single-level index.
            if not isinstance(idx_a, tuple):
                idx_a = (idx_a,)
            if not isinstance(idx_b, tuple):
                idx_b = (idx_b,)

            index_tuples.append(idx_a + idx_b)
            correlations.append(corr)

    # Create the MultiIndex and correlation DataFrame.
    index = pd.MultiIndex.from_tuples(index_tuples, names=combined_idx_names)
    df_corr = pd.DataFrame({"correlation": correlations}, index=index)

    return df_corr