import functools
import math
import numpy as np
import numpy as np
import os
import pandas as pd
from anndata import AnnData
from decoupler._Method import Method, MethodMeta
from decoupler._log import _log
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy.stats import zscore, norm
from tqdm.auto import tqdm

# CORES_USED = 1 # For debugging, use a single core
CORES_USED = max(1, int(os.cpu_count() * 0.8))  # Use 80% of available cores


def std_dev_mean_norm_rank(n_population: int, k_sample: int) -> float:
    """Calculates the standard deviation of the mean of k ranks drawn from n (unweighted)."""
    if not (isinstance(n_population, int) and n_population > 0):
        raise ValueError("n_population must be a positive integer.")
    if not (isinstance(k_sample, int) and k_sample > 0):
        raise ValueError("k_sample must be a positive integer.")
    if k_sample > n_population:
        raise ValueError("Sample size k_sample cannot exceed population size n_population.")

    if k_sample == n_population or n_population == 1:
        return 0.0

    var = ((n_population + 1) * (n_population - k_sample)) / (
        12 * n_population ** 2 * k_sample
    )
    return math.sqrt(max(var, 0.0))


def std_dev_weighted_mean_norm_rank(n_population: int, weights: np.ndarray) -> float:
    """Calculates the standard deviation of the weighted mean of k ranks drawn from n."""
    if not (isinstance(n_population, int) and n_population > 0):
        raise ValueError("n_population must be a positive integer.")
    if not isinstance(weights, np.ndarray) or weights.size == 0:
        raise ValueError("weights must be a non-empty numpy array.")

    if n_population == 1:
        return 0.0

    sum_sq_weights = np.sum(np.square(weights))
    numerator = (n_population + 1) * (n_population * sum_sq_weights - 1)
    denominator = 12 * n_population**2

    if denominator == 0:
        return 0.0

    var = numerator / denominator
    return math.sqrt(max(var, 0.0))



def _process_single_cell_unweighted_ranks(
    cell_name: str, expression_series: pd.Series, priors_df: pd.DataFrame
) -> tuple[str, list, list, list]:
    """Processes a single cell using the UNWEIGHTED mean rank method."""
    expr = expression_series.dropna().sort_values(ascending=False)
    n_genes = expr.size

    if n_genes == 0:
        return cell_name, [], [], []

    ranks = (np.arange(1, n_genes + 1) - 0.5) / n_genes
    rank_df = pd.DataFrame({"target": expr.index, "Rank": ranks})

    p = priors_df.merge(rank_df, on="target", how="left").dropna(subset=["Rank"])
    p["AdjustedRank"] = np.where(p["weight"] < 0, 1 - p["Rank"], p["Rank"])

    tf_summary = (
        p.groupby("source", observed=True)
        .agg(
            AvailableTargets=("AdjustedRank", "size"),
            RankMean=("AdjustedRank", "mean"),
        )
        .reset_index()
    )

    tf_summary = tf_summary[tf_summary["AvailableTargets"] > 0]
    if tf_summary.empty:
        return cell_name, [], [], []

    tf_summary["ActivationDir"] = np.where(tf_summary["RankMean"] < 0.5, 1, -1)
    tf_summary["Sigma_n_k"] = tf_summary["AvailableTargets"].apply(
        lambda k: std_dev_mean_norm_rank(n_genes, k)
    )
    tf_summary["Z"] = (tf_summary["RankMean"] - 0.5) / tf_summary["Sigma_n_k"].replace(
        0, np.nan
    )
    tf_summary["P_two_tailed"] = np.where(
        tf_summary["RankMean"] < 0.5,
        2 * norm.cdf(tf_summary["Z"]),
        2 * (1 - norm.cdf(tf_summary["Z"])),
    )

    return (
        cell_name,
        tf_summary["source"].tolist(),
        tf_summary["P_two_tailed"].tolist(),
        tf_summary["ActivationDir"].tolist(),
    )


def _process_single_cell_weighted_ranks(
    cell_name: str, expression_series: pd.Series, priors_df: pd.DataFrame
) -> tuple[str, list, list, list]:
    """Processes a single cell using the WEIGHTED mean rank method."""
    expr = expression_series.dropna().sort_values(ascending=False)
    n_genes = expr.size

    if n_genes == 0:
        return cell_name, [], [], []

    ranks = (np.arange(1, n_genes + 1) - 0.5) / n_genes
    rank_df = pd.DataFrame({"target": expr.index, "Rank": ranks})

    p = priors_df.merge(rank_df, on="target", how="left").dropna(subset=["Rank"])
    p["abs_weight"] = p["weight"].abs()
    p["AdjustedRank"] = np.where(p["weight"] < 0, 1 - p["Rank"], p["Rank"])

    # Use .apply with a lambda to perform the complex weighted aggregation
    tf_summary = (
        p.groupby("source", observed=True)
        .apply(
            lambda grp: pd.Series(
                {
                    "AvailableTargets": len(grp),
                    "RankMean": np.average(
                        grp["AdjustedRank"],
                        weights=grp["abs_weight"] / grp["abs_weight"].sum(),
                    ),
                    "Sigma_n_k": std_dev_weighted_mean_norm_rank(
                        n_genes, (grp["abs_weight"] / grp["abs_weight"].sum()).values
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    tf_summary = tf_summary[tf_summary["AvailableTargets"] > 0]
    if tf_summary.empty:
        return cell_name, [], [], []

    tf_summary["ActivationDir"] = np.where(tf_summary["RankMean"] < 0.5, 1, -1)
    tf_summary["Z"] = (tf_summary["RankMean"] - 0.5) / tf_summary["Sigma_n_k"].replace(
        0, np.nan
    )
    tf_summary["P_two_tailed"] = np.where(
        tf_summary["RankMean"] < 0.5,
        2 * norm.cdf(tf_summary["Z"]),
        2 * (1 - norm.cdf(tf_summary["Z"])),
    )

    return (
        cell_name,
        tf_summary["source"].tolist(),
        tf_summary["P_two_tailed"].tolist(),
        tf_summary["ActivationDir"].tolist(),
    )


def _process_single_cell_stouffer(
    cell_name: str,
    expression_z_scores: pd.Series,
    priors_df: pd.DataFrame
) -> tuple[str, list, list, list]:
    priors_group = priors_df.groupby("source", observed=True).agg(
        {"target": list, "weight": list}
    )

    valid_z_scores = expression_z_scores.dropna()
    if valid_z_scores.empty:
        return cell_name, [], [], []

    z_score_map = valid_z_scores.to_dict()
    regulators, p_values, directions = [], [], []

    # Iterate through the pre-grouped priors
    for tf, data in priors_group.iterrows():
        target_genes = data["target"]
        effects = data["weight"]

        evidence_z = []
        for i, gene in enumerate(target_genes):
            z = z_score_map.get(gene)
            if z is not None:
                evidence_z.append(-z if effects[i] < 0 else z)

        k = len(evidence_z)

        # Calculate Stouffer's Z
        z_sum = np.sum(evidence_z)
        stouffer_z = z_sum / np.sqrt(k) if k > 0 else 0

        # Use a two-tailed test for the p-value
        p_value = 2 * norm.sf(abs(stouffer_z))

        # Direction is the sign of the combined evidence
        direction = np.sign(stouffer_z)

        regulators.append(tf)
        p_values.append(p_value)
        directions.append(direction)
    return cell_name, regulators, p_values, directions


def run_tf_analysis(
    adata: AnnData,
    priors: pd.DataFrame,
    ignore_zeros: bool,
    min_targets: int,
    analysis_method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if adata is None or adata.n_obs == 0:
            raise ValueError("Input data is empty or invalid")

        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()

        # Filter priors based on min_targets and presence in the data
        priors = priors[priors["target"].isin(adata.var_names)]
        priors = priors.groupby("source").filter(lambda x: len(x) >= min_targets)

        process_func = None
        data_for_processing = None

        # Step 1: Prepare the correct input data matrix based on the analysis method
        if ("zscore" in analysis_method):  # For 'ranks_from_zscore' and 'stouffers_zscore'
            print("Calculating Z-scores as required by the selected method...")
            if ignore_zeros:
                X[X == 0] = np.nan
                z_mat = zscore(X, axis=0, nan_policy="omit")
            else:
                z_mat = zscore(X, axis=0)
            data_for_processing = pd.DataFrame(
                z_mat, index=adata.obs_names, columns=adata.var_names
            )

        elif "ranks_of_ranks" in analysis_method:
            print("Using raw gene expression as input for per-cell ranking...")
            data_for_processing = pd.DataFrame(
                X, index=adata.obs_names, columns=adata.var_names
            )
            if ignore_zeros:
                # The processing functions will handle NaNs correctly during ranking
                data_for_processing.replace(0, np.nan, inplace=True)
        else:
            raise ValueError(f"Unknown analysis method family for: {analysis_method}")

        # Step 2: Select the appropriate worker function
        if analysis_method in ["ranks_from_zscore", "ranks_of_ranks"]:
            print("-> Using UNWEIGHTED mean rank calculation.")
            process_func = _process_single_cell_unweighted_ranks

        elif analysis_method in [
            "weighted_ranks_from_zscore",
            "weighted_ranks_of_ranks",
        ]:
            print("-> Using WEIGHTED mean rank calculation.")
            process_func = _process_single_cell_weighted_ranks

        elif analysis_method == "stouffers_zscore":
            print("-> Using Stouffer's Z-score method.")
            process_func = _process_single_cell_stouffer

        else:
            raise ValueError(f"Unknown analysis method: {analysis_method}")

        if process_func is None or data_for_processing is None:
            raise RuntimeError(
                "Failed to select a processing function or prepare data."
            )

        # ───── Parallel processing of cells ───────────────────────────────────────
        print(f"Starting TF activity using {CORES_USED if CORES_USED > 0 else 'all available'} cores.")
        tasks = [
            delayed(process_func)(
                cell_name,
                data_for_processing.loc[cell_name],
                priors,
            )
            for cell_name in data_for_processing.index
        ]

        if CORES_USED == 1:
            print("Running sequentially (CORES_USED=1)...")
            cell_results_list = [
                process_func(cell_name, data_for_processing.loc[cell_name], priors)
                for cell_name in tqdm(data_for_processing.index, desc="Processing cells in sequence")
            ]
        else:
            print(f"Running in parallel with CORES_USED={CORES_USED}.")
            cell_results_list = Parallel(n_jobs=CORES_USED, backend="loky", verbose=1)(
                tqdm(tasks, desc="Processing cells in parallel", total=len(tasks))
            )

        # ───── Aggregate results into two separate DataFrames ───────────────────
        print("\nAggregating results...")
        records = [
            (cell, reg, pval, direc)
            for cell, regs, pvals, direcs in cell_results_list
            for reg, pval, direc in zip(regs, pvals, direcs)
        ]

        if not records:
            print(
                "Warning: No TF activities could be computed. Returning empty DataFrame."
            )
            return pd.DataFrame(index=adata.obs_names), pd.DataFrame(
                index=adata.obs_names
            )

        results_df = pd.DataFrame(
            records, columns=["cell", "regulator", "p_value", "direction"]
        )
        pvalue_df = results_df.pivot(
            index="cell", columns="regulator", values="p_value"
        )
        activation_df = results_df.pivot(
            index="cell", columns="regulator", values="direction"
        )

        pvalue_df = pvalue_df.reindex(adata.obs_names)
        activation_df = activation_df.reindex(adata.obs_names)

        pvalue_df.dropna(axis=1, how="all", inplace=True)
        scores = -np.log10(pvalue_df).multiply(activation_df, fill_value=0)

        print("kale completed")
        return scores, pvalue_df

    except Exception as e:
        print(f"Error during TF analysis: {e}")
        raise


def _func_kale(
    mat: np.ndarray or None = None,
    adj: np.ndarray or None = None,
    adata: AnnData = None,
    net: pd.DataFrame = None,
    verbose: bool = False,
    method: str = "ranks_from_zscore",
    min_targets: int = 0,
    ignore_zeros: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Kale transcription factor activity inference method.

    Note: `mat` and `adj` are unused but included for decoupler framework compatibility.
          The method operates directly on the `adata` and `net` inputs.

    Args:
        mat: Not used, present for compatibility.
        adj: Not used, present for compatibility.
        adata (anndata.AnnData): An AnnData object with expression data.
        net (pd.DataFrame): A DataFrame with regulator-target interactions.
                            Must have 'source', 'target', and 'weight' columns.
        method (str, optional): The analysis method to use. Options are:
            - 'ranks_from_zscore': Unweighted mean rank on Z-scores. (Default)
            - 'weighted_ranks_from_zscore': Weighted mean rank on Z-scores.
            - 'ranks_of_ranks': Unweighted ranks of ranks on raw expression.
            - 'weighted_ranks_of_ranks': Weighted ranks of ranks on raw expression.
            - 'stouffers_zscore': Stouffer's method on Z-scores.
        min_targets (int, optional): Minimum number of targets for a regulator. Defaults to 0.
        ignore_zeros (bool, optional): If True, treat zeros as missing values. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - scores (pd.DataFrame): A matrix of TF activity scores (cells x TFs).
            - pvalue_df (pd.DataFrame): A matrix of p-values (cells x TFs).
    """

    # Check that adata and net are provided, since they are now the primary inputs
    if adata is None or net is None:
        raise ValueError("`adata` and `net` must be provided for the kale method.")

    if "weighted" not in method:
        _log(
            "Not Weighted method selected; converting all weights to 1 or -1.",
            level="warning",
            verbose=verbose,
        )
        net["weight"] = np.where(net["weight"] > 0, 1, -1)

    scores, pvalues = run_tf_analysis(adata, net, ignore_zeros, min_targets=0, analysis_method=method)

    # Ensure final output aligns with all sources in the net, filling missing with NaN
    all_sources = net["source"].unique()
    scores = scores.reindex(columns=all_sources, fill_value=np.nan)
    pvalues = pvalues.reindex(columns=all_sources, fill_value=np.nan)

    return scores.to_numpy(), pvalues.to_numpy()


_kale = MethodMeta(
    name="kale",
    desc="Cell Activity Inference Landscape (KALE)",
    func=_func_kale,
    stype="numerical",
    adj=True,
    weight=True,
    test=False,
    limits=(-np.inf, +np.inf),
    reference="",
)
kale = Method(_method=_kale)
