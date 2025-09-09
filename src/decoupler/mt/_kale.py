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
    """
    Calculates the standard deviation of the weighted mean of k ranks drawn from n.
    Assumes weights are normalized to sum to 1.
    """
    if not (isinstance(n_population, int) and n_population > 0):
        raise ValueError("n_population must be a positive integer.")
    if not isinstance(weights, np.ndarray) or weights.size == 0:
        raise ValueError("weights must be a non-empty numpy array.")

    if n_population == 1:
        return 0.0

    # Formula: sqrt( (n+1) * (n * sum(w_i^2) - 1) / (12 * n^2) )
    sum_sq_weights = np.sum(np.square(weights))
    numerator = (n_population + 1) * (n_population * sum_sq_weights - 1)
    denominator = 12 * n_population ** 2

    if denominator == 0:
        return 0.0

    var = numerator / denominator
    return math.sqrt(max(var, 0.0))


def _process_single_cell_ranks(
    cell_name: str,
    expression_series: pd.Series,
    priors_df: pd.DataFrame,
    use_weights: bool,
) -> tuple[str, list, list, list]:
    """
    Processes a single cell to calculate TF activity scores using rank-based methods.
    Can perform either standard (unweighted) or weighted mean rank analysis.
    """
    expr = expression_series.dropna().sort_values(ascending=False)
    n_genes = expr.size

    if n_genes == 0:
        return cell_name, {}

    ranks = (np.arange(1, n_genes + 1) - 0.5) / n_genes
    rank_df = pd.DataFrame({"target": expr.index, "rank": ranks})

    p = priors_df.merge(rank_df, on="target", how="left")
    p = p[p["rank"].notna()]

    p["abs_weight"] = p["weight"].abs()
    p["adjusted_rank"] = np.where(p["weight"] < 0, 1 - p["rank"], p["rank"])

    if p.empty:
        return cell_name, [], [], []

    # Helper function to calculate stats for each TF group
    def _calculate_stats_for_tf(group):
        k = len(group)
        if use_weights:
            # Normalize weights for this specific set of available targets to sum to 1
            weights = group["abs_weight"].values
            norm_weights = weights / np.sum(weights)

            # Calculate weighted mean rank
            mean_rank = np.average(group["adjusted_rank"], weights=norm_weights)
            sigma = std_dev_weighted_mean_norm_rank(n_genes, norm_weights)
        else:
            # Calculate standard mean rank
            mean_rank = group["adjusted_rank"].mean()
            sigma = std_dev_mean_norm_rank(n_genes, k)

        return pd.Series(
            {"available_targets": k, "rank_mean": mean_rank, "sigma_n_k": sigma}
        )

    tf_summary = (
        p.groupby("source", observed=True).apply(_calculate_stats_for_tf).reset_index()
    )
    tf_summary = tf_summary[tf_summary["available_targets"] > 0]

    if tf_summary.empty:
        return cell_name, [], [], []

    tf_summary["acti_dir"] = np.where(tf_summary["rank_mean"] < 0.5, 1, -1)
    tf_summary["Z"] = (tf_summary["rank_mean"] - 0.5) / (
        tf_summary["sigma_n_k"].replace(0, np.nan)
    )
    tf_summary["p_two_tailed"] = np.where(
        tf_summary["rank_mean"] < 0.5,
        2 * norm.cdf(tf_summary["Z"]),
        2 * (1 - norm.cdf(tf_summary["Z"])),
    )

    # Handle cases where Z might be NaN (if sigma is 0)
    tf_summary["p_two_tailed"] = tf_summary["p_two_tailed"].fillna(1.0)

    regulators = tf_summary["source"].tolist()
    p_values = tf_summary["p_two_tailed"].tolist()
    directions = tf_summary["acti_dir"].tolist()

    return cell_name, regulators, p_values, directions


def _process_single_cell_stouffer(
    cell_name: str,
    expression_z_scores: pd.Series,
    priors_df: pd.DataFrame
) -> tuple[str, list, list, list]:
    priors_group = (
        priors_df.groupby("source", observed=True)
        .agg({
            "target": list,
            "weight": list
        })
    )

    valid_z_scores = expression_z_scores.dropna()
    if valid_z_scores.empty:
        return cell_name, [], [], []

    z_score_map = valid_z_scores.to_dict()

    regulators = []
    p_values = []
    directions = []

    for tf, data in priors_group.iterrows():
        target_genes = data["target"]
        effects = data["weight"]
        evidence_z = []
        for i, gene in enumerate(target_genes):
            z = z_score_map.get(gene)
            if z is not None:
                evidence_z.append(-z if effects[i] < 0 else z)

        k = len(evidence_z)
        if k == 0:
            continue

        z_sum = np.sum(evidence_z)
        stouffer_z = z_sum / np.sqrt(k)
        p_value = 2 * norm.sf(abs(stouffer_z))
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

        print("Calculating Z-scores...")
        if ignore_zeros:
            print("Calculating Z-scores based only on non-zero expression values.")
            X[X == 0] = np.nan
            X[X == 0.0] = np.nan
            z_mat = zscore(X, axis=0, nan_policy="omit")  # across all cells for a single gene (axis=0)
        else:
            print("Calculating standard Z-scores including zero values.")
            z_mat = zscore(X, axis=0)

        z_df = pd.DataFrame(z_mat, index=adata.obs_names, columns=adata.var_names)

        print("Loading TF-target priors...")
        priors = priors.groupby("source").filter(lambda x: len(x) >= min_targets)
        priors = priors[priors["target"].isin(z_df.columns)]
        priors_group = priors.groupby("source", observed=True).agg(
            {"weight": list, "target": list}
        )
        priors_group = priors_group[priors_group["weight"].apply(len) >= min_targets]
        priors = priors[priors["source"].isin(priors_group.index)]

        # ───── Decide the analysis method ───────────────────────────────────────
        if analysis_method in ["ranks_from_zscore", "weighted_ranks_from_zscore"]:
            print(
                f"Using Rank-Based method from Z-scores. Weighted: {'weighted' in analysis_method}."
            )
            use_weights = "weighted" in analysis_method
            process_func = functools.partial(
                _process_single_cell_ranks, use_weights=use_weights
            )
            data_for_processing = z_df

        elif analysis_method == "stouffers_zscore":
            print("Using Stouffer's Z-score method for inference.")
            process_func = _process_single_cell_stouffer
            data_for_processing = z_df

        elif analysis_method in [
            "ranks_from_gene_expression",
            "weighted_ranks_from_gene_expression",
        ]:
            print(f"Using Rank-Based method from Gene Expression. Weighted: {'weighted' in analysis_method}.")
            raw_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
            rank_input_df = raw_df.copy()
            rank_input_df[rank_input_df == 0] = np.nan
            ranked_across_cells = rank_input_df.rank(axis=0, na_option="keep")
            ranked_across_cells = (ranked_across_cells - 0.5) / ranked_across_cells.max(
                axis=0, skipna=True
            )

            use_weights = "weighted" in analysis_method
            process_func = functools.partial(
                _process_single_cell_ranks, use_weights=use_weights
            )
            data_for_processing = ranked_across_cells

        else:
            raise ValueError(f"Unknown analysis method: {analysis_method}")

        print(f"Starting TF activity using {CORES_USED if CORES_USED > 0 else 'all available'} cores.")
        tasks = [
            delayed(process_func)(
                cell_name,
                data_for_processing.loc[cell_name],
                priors,
            )
            for cell_name in data_for_processing.index
        ]

        if CORES_USED == 1:  # Useful for debugging
            print("Running sequentially (CORES_USED=1)...")
            cell_results_list = [
                process_func(cell_name, data_for_processing.loc[cell_name], priors)
                for cell_name in tqdm(data_for_processing.index, desc="Processing cells")
            ]
        else:
            print(f"Running in parallel with CORES_USED={CORES_USED}.")
            cell_results_list = Parallel(n_jobs=CORES_USED, backend="loky", verbose=0)(
                tqdm(tasks, desc="Processing cells")
            )

        print("\nAggregating results...")
        p_value_records = []
        activation_records = []

        # 1. Unpack results into flat lists of records
        for cell_name, regulators, p_values, directions in tqdm(cell_results_list, desc="Unpacking results"):
            if not regulators:
                continue  # Skip if no results for this cell
            for i, regulator in enumerate(regulators):
                p_value_records.append({'cell': cell_name, 'regulator': regulator, 'p_value': p_values[i]})
                activation_records.append({'cell': cell_name, 'regulator': regulator, 'direction': directions[i]})

        if not p_value_records:
            print("Warning: No TF activities could be computed. Returning empty DataFrame.")
            return pd.DataFrame(index=adata.obs_names), pd.DataFrame(index=adata.obs_names)

        # 2. Build temporary DataFrames from the lists of records (very fast)
        p_values_temp_df = pd.DataFrame.from_records(p_value_records)
        activation_temp_df = pd.DataFrame.from_records(activation_records)

        # 3. Pivot the temporary DataFrames into the desired final shape (cells x regulators)
        pvalue_df = p_values_temp_df.pivot(index='cell', columns='regulator', values='p_value')
        activation_df = activation_temp_df.pivot(index='cell', columns='regulator', values='direction')

        pvalue_df = pvalue_df.reindex(adata.obs_names)
        activation_df = activation_df.reindex(adata.obs_names)

        scores = -np.log10(pvalue_df)
        scores = scores.multiply(activation_df, fill_value=0)

        print("kale completed")
        return scores, pvalue_df

    except Exception as e:
        print(f"Error: {e}")
        raise e


def _func_kale(
    mat: np.ndarray,
    adj: np.ndarray,
    adata: AnnData,
    net: pd.DataFrame,
    verbose: bool = False,
    method: str = "ranks_from_zscore",
    n_targets: int = 0,
    ignore_zeros: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    if "weighted" in method and (adj < 0).any():
        _log(
            "Negative weights found in the network. Switching to unweighted rank-based method.",
            level="warning",
            verbose=verbose,
        )
        net["weight"] = np.where(net["weight"] > 0, 1, -1)

    scores, pvalues = run_tf_analysis(adata, net, ignore_zeros, min_targets=0, analysis_method=method)

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
