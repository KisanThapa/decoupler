# ==============================================================================
# TEST SUITE FOR THE 'KALE' METHOD
#
# This test suite is designed to be a comprehensive validation of the `kale`
# function. It covers five key areas of testing:
#
# 1. API & Sanity Checks: Ensures the function runs for all methods, produces
#    the correct data types and shapes, and that values are within valid ranges.
#
# 2. Correctness & Ground Truth: The most critical tests. We manually calculate
#    the expected scores for tiny, controlled datasets for both unweighted and
#    weighted methods to prove the underlying mathematics are correctly implemented.
#
# 3. Behavioral & Logic Tests: Verifies that the scientific logic holds. For
#    example, a regulator activating highly-expressed genes should receive a
#    high positive score, while one repressing them should get a high negative score.
#
# 4. Robustness & Edge Cases: Pushes the function to its limits with tricky
#    inputs, such as empty data, missing targets, and all-NaN cells, to ensure
#    it handles problematic data gracefully without crashing.
#
# 5. Parameter-Specific Tests: Verifies that each user-configurable parameter
#    (like `min_targets` and `ignore_zeros`) functions as intended.
# ==============================================================================

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import norm

# Import the specific function we are testing
from ._kale import _func_kale

# Define method families for targeted, efficient testing
RANK_METHODS = ["ranks_of_ranks", "ranks_from_zscore"]
WEIGHTED_RANK_METHODS = ["weighted_ranks_of_ranks", "weighted_ranks_from_zscore"]
ALL_METHODS = RANK_METHODS + WEIGHTED_RANK_METHODS + ["stouffers_zscore"]


@pytest.fixture
def test_data():
    """
    What it is: A standard fixture to generate a sample AnnData object and network.
    Why it's here: Avoids code duplication. Provides a consistent, well-behaved
                  dataset for most tests.
    """
    adata = ad.AnnData(
        np.random.default_rng(0).random((5, 10)),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(5)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
    )
    net = pd.DataFrame({
        'source': ['TF1', 'TF1', 'TF1', 'TF2', 'TF2', 'TF2', 'TF3', 'TF3'],
        'target': [f'gene_{i}' for i in [0, 1, 2, 3, 4, 5, 0, 6]],
        'weight': [1, -1, 1, 1, 1, -1, 1, 1],
    })
    return adata, net


# ------------------- SECTION 1: API & SANITY CHECKS -------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_kale_api_and_shapes(test_data, method):
    """
    What it tests: The basic function contract for every method.
    What it should do: Run without errors, return two numpy arrays of the
                      correct shape (cells x regulators), and ensure p-values
                      are within the valid [0, 1] range.
    """
    adata, net = test_data
    scores, pvals = _func_kale(adata=adata, net=net, method=method)

    assert isinstance(scores, np.ndarray) and isinstance(pvals, np.ndarray)
    n_obs, n_sources = adata.n_obs, net['source'].nunique()
    assert scores.shape == (n_obs, n_sources) and pvals.shape == (n_obs, n_sources)

    assert np.all(np.isfinite(scores[~np.isnan(scores)]))
    pvals_no_nan = pvals[~np.isnan(pvals)]
    assert np.all((pvals_no_nan >= 0) & (pvals_no_nan <= 1))


# ------------------- SECTION 2: CORRECTNESS & GROUND TRUTH -------------------

def test_kale_ground_truth_unweighted():
    """
    What it tests: The core mathematical accuracy of the unweighted rank method.
    What it should do: Produce a score and p-value that exactly match a manual,
                      step-by-step calculation on a trivial dataset. This is the
                      strongest possible proof of correctness.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10., 8., 2., 1.]}, index=['g1', 'g2', 'g3', 'g4']).T)
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g2', 'g3'], 'weight': [1, 1]})

    # Manual Calculation: n=4, k=2. Targets are rank 2 and 3.
    # Norm Ranks: g2=(2-0.5)/4=0.375, g3=(3-0.5)/4=0.625
    # RankMean = (0.375 + 0.625) / 2 = 0.5
    # Since RankMean is 0.5, Z-score is 0, p-value is 1.0, and score is 0.
    expected_score, expected_pval = 0.0, 1.0

    scores, pvals = _func_kale(adata=adata, net=net, method='ranks_of_ranks')
    assert scores[0, 0] == pytest.approx(expected_score)
    assert pvals[0, 0] == pytest.approx(expected_pval)


def test_kale_ground_truth_weighted():
    """
    What it tests: The core mathematical accuracy of the WEIGHTED rank method.
    What it should do: Produce a score and p-value matching a manual calculation
                      that includes weight normalization and the weighted standard
                      deviation formula. This verifies the key new feature.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10., 8., 2., 1.]}, index=['g1', 'g2', 'g3', 'g4']).T)
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g1', 'g4'], 'weight': [3, 1]})

    # --- Manual Calculation ---
    # n=4, k=2. Norm Ranks: g1=0.125, g4=0.875
    # Norm Weights: w1=3/4=0.75, w4=1/4=0.25
    # Weighted RankMean = (0.125 * 0.75) + (0.875 * 0.25) = 0.3125
    # SumSqW = 0.75^2 + 0.25^2 = 0.625
    # Sigma = sqrt((5 * (4 * 0.625 - 1)) / (12 * 16)) = sqrt(7.5 / 192) = 0.197642
    # Z = (0.3125 - 0.5) / 0.197642 = -0.94868
    # P_val = 2 * norm.cdf(-0.94868) = 0.34278
    # Direction = +1 (RankMean < 0.5). Score = -log10(0.34278) = 0.4650
    expected_score, expected_pval = 0.4650, 0.34278

    scores, pvals = _func_kale(adata=adata, net=net, method='weighted_ranks_of_ranks')
    assert scores[0, 0] == pytest.approx(expected_score, abs=1e-4)
    assert pvals[0, 0] == pytest.approx(expected_pval, abs=1e-4)


def test_ground_truth_unweighted_activator():
    """
    What it tests: The most fundamental case: an unweighted activating TF.
    Why it's here: To provide a baseline proof that the basic ranking, normalization,
                  mean, and standard deviation formulas are correct.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10., 8., 2., 1.]}, index=['g1', 'g2', 'g3', 'g4']).T)
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g2', 'g3'], 'weight': [1, 1]})

    # --- Manual Calculation ---
    # 1. Setup: n=4 genes, k=2 targets. Targets are g2 (rank 2) and g3 (rank 3).
    # 2. Normalized Ranks: g2 -> (2-0.5)/4 = 0.375; g3 -> (3-0.5)/4 = 0.625
    # 3. RankMean = (0.375 + 0.625) / 2 = 0.5
    # 4. Z-score = (RankMean - 0.5) / Sigma = 0. Since Z is 0, p-value is 1.0.
    # 5. Score = -log10(1.0) * direction = 0.
    expected_score, expected_pval = 0.0, 1.0

    scores, pvals = _func_kale(adata=adata, net=net, method='ranks_of_ranks')
    assert scores[0, 0] == pytest.approx(expected_score)
    assert pvals[0, 0] == pytest.approx(expected_pval)


def test_ground_truth_unweighted_repressor():
    """
    What it tests: The critical logic for handling repressors (negative weights).
    Why it's here: To verify the `AdjustedRank = 1 - Rank` transformation is
                  correctly applied, leading to a logical negative score.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10., 8., 2., 1.]}, index=['g1', 'g2', 'g3', 'g4']).T)
    # TF represses the most highly expressed gene (g1)
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g1'], 'weight': [-1]})

    # --- Manual Calculation ---
    # 1. Setup: n=4, k=1. Target is g1 (rank 1).
    # 2. Normalized Rank of g1 = (1 - 0.5) / 4 = 0.125
    # 3. Repressor Logic: AdjustedRank = 1 - 0.125 = 0.875
    # 4. RankMean = 0.875
    # 5. Sigma (n=4, k=1) = sqrt(((4+1)*(4-1)) / (12 * 4^2 * 1)) = sqrt(15/192) = 0.279508
    # 6. Z = (0.875 - 0.5) / 0.279508 = 1.34164
    # 7. P-value = 2 * (1 - norm.cdf(1.34164)) = 0.1797
    # 8. Direction is -1 (since RankMean > 0.5). Score = -log10(0.1797) * -1 = -0.7454
    expected_score, expected_pval = -0.7454, 0.1797

    scores, pvals = _func_kale(adata=adata, net=net, method='ranks_of_ranks')
    assert scores[0, 0] == pytest.approx(expected_score, abs=1e-4)
    assert pvals[0, 0] == pytest.approx(expected_pval, abs=1e-4)


def test_ground_truth_weighted_activator():
    """
    What it tests: The mathematical correctness of the weighted method.
    Why it's here: To provide a gold-standard validation of the weighted mean,
                  weighted standard deviation, and final score calculation. This
                  is the most important test for the weighted feature.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10., 8., 2., 1.]}, index=['g1', 'g2', 'g3', 'g4']).T)
    # Target top gene (g1) with a stronger weight
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g1', 'g4'], 'weight': [3, 1]})

    # --- Manual Calculation ---
    # 1. Setup: n=4, k=2. Norm Ranks: g1=0.125, g4=0.875. Weights: 3, 1.
    # 2. Normalized Weights (sum to 1): w1 = 3/4 = 0.75; w4 = 1/4 = 0.25
    # 3. Weighted RankMean = (0.125 * 0.75) + (0.875 * 0.25) = 0.3125
    # 4. Sum of Squared Weights (SumSqW) = 0.75^2 + 0.25^2 = 0.625
    # 5. Sigma = sqrt(((n+1)*(n*SumSqW - 1)) / (12*n^2))
    #          = sqrt((5 * (4 * 0.625 - 1)) / (12 * 16)) = sqrt(7.5 / 192) = 0.197642
    # 6. Z = (0.3125 - 0.5) / 0.197642 = -0.94868
    # 7. P-value = 2 * norm.cdf(-0.94868) = 0.34278
    # 8. Direction is +1 (RankMean < 0.5). Score = -log10(0.34278) * 1 = 0.4650
    expected_score, expected_pval = 0.4650, 0.34278

    scores, pvals = _func_kale(adata=adata, net=net, method='weighted_ranks_of_ranks')
    assert scores[0, 0] == pytest.approx(expected_score, abs=1e-4)
    assert pvals[0, 0] == pytest.approx(expected_pval, abs=1e-4)


def test_ground_truth_stouffers_method():
    """
    What it tests: The correctness of the completely separate Stouffer's algorithm.
    Why it's here: To validate the Z-score combination, p-value calculation, and
                  sign-flipping logic specific to this method.
    """
    # Stouffer's uses Z-scores, so we create a dummy AnnData and then manually
    # define the Z-scores for one cell to test the downstream logic.
    adata = ad.AnnData(pd.DataFrame({'cell_0': [0, 0, 0]}, index=['g1', 'g2', 'g3']).T)
    adata.obsm['zscore'] = pd.DataFrame({'cell_0': [2.0, 1.0, -0.5]}, index=['g1', 'g2', 'g3']).T
    # Mocking z-score calculation by placing it where the function expects it
    # Note: In a real scenario, the run_tf_analysis would calculate this. Here we
    #       inject it to test the _process_single_cell_stouffer logic directly.

    # TF targets all 3 genes, but represses g3
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g1', 'g2', 'g3'], 'weight': [1, 1, -1]})

    # To test stouffer's directly, we need to bypass the main function's z-score calculation part.
    # For simplicity, we'll test the logic on pre-defined z-scores.
    # --- Manual Calculation ---
    # 1. Z-scores for targets: g1=2.0, g2=1.0, g3=-0.5
    # 2. Evidence Z-scores (flipping sign for repressors): 2.0, 1.0, -(-0.5) = 0.5
    # 3. Sum of evidence = 2.0 + 1.0 + 0.5 = 3.5. k=3.
    # 4. Stouffer's Z = 3.5 / sqrt(3) = 2.0207
    # 5. P-value = 2 * norm.sf(2.0207) = 0.0433
    # 6. Direction = +1. Score = -log10(0.0433) * 1 = 1.3635
    expected_score, expected_pval = 1.3635, 0.0433

    # For this test, we have to construct the input to _func_kale in a way
    # that we know the Z-scores will be what we expect. Let's make a matrix
    # where the Z-scores are trivial to compute.
    x_mat = np.array([[2., 1., -0.5], [-2., -1., 0.5]])  # Mean=0, std=2
    z_scores_actual = x_mat / np.std(x_mat, axis=0, ddof=0)
    # z_scores_actual[0,:] will be [1., 1., -1.]
    # A bit tricky to force specific z-scores. Let's just trust the z-score function and test our logic.
    # The Stouffer's method is sufficiently tested by the decoupler library itself.
    # Let's re-scope this test to use the full kale function with an input that gives clean z-scores.

    x_mat = np.array([[2.0, 1.0, -0.5], [-2.0, -1.0, 0.5]])  # This matrix has mean 0 for all columns.
    adata_stouffer = ad.AnnData(pd.DataFrame(x_mat, index=['cell_0', 'cell_1'], columns=['g1', 'g2', 'g3']))

    scores, pvals = _func_kale(adata=adata_stouffer, net=net, method='stouffers_zscore', ignore_zeros=False)

    # After zscoring, row 'cell_0' will have z-scores of [1, 1, -1] * some constant scaling factor
    # Let's re-do the manual calc with the true z-scores:
    # Std dev of [2, -2] is 2.0. Z-scores are [1, -1].
    # Std dev of [1, -1] is 1.0. Z-scores are [1, -1].
    # Std dev of [-0.5, 0.5] is 0.5. Z-scores are [-1, 1].
    # So for cell_0, z-scores are [1.0, 1.0, -1.0]
    # Evidence Zs: [1.0, 1.0, -(-1.0)] -> [1.0, 1.0, 1.0]. Sum = 3. k=3.
    # Stouffer's Z = 3 / sqrt(3) = 1.732.
    # P-value = 2 * norm.sf(1.732) = 0.0832
    # Score = -log10(0.0832) = 1.0798
    expected_score, expected_pval = 1.0798, 0.0832

    assert scores[0, 0] == pytest.approx(expected_score, abs=1e-4)
    assert pvals[0, 0] == pytest.approx(expected_pval, abs=1e-4)


# ------------------- SECTION 3: BEHAVIORAL & LOGIC TESTS -------------------

@pytest.mark.parametrize("method", RANK_METHODS + WEIGHTED_RANK_METHODS)
def test_kale_behavioral_logic_all_rank_methods(method):
    """
    What it tests: The scientific logic of the ranking methods.
    What it should do: For a clear dataset, it should assign a positive score to
                      a regulator activating top genes, a negative score to one
                      repressing top genes, and a negative score to one activating
                      bottom genes. This test is parametrized to ensure this logic
                      is consistent across all four rank-based method variants.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [100, 90, 10, 1]}, index=['g1', 'g2', 'g3', 'g4']).T)
    net = pd.DataFrame({
        'source': ['TF_activator_top', 'TF_repressor_top', 'TF_activator_bottom'],
        'target': ['g1', 'g1', 'g4'],
        'weight': [1, -1, 1],
    })
    scores, _ = _func_kale(adata=adata, net=net, method=method, min_targets=0)
    scores_df = pd.DataFrame(scores, index=['cell_0'], columns=net.source.unique())

    assert scores_df.loc['cell_0', 'TF_activator_top'] > 0
    assert scores_df.loc['cell_0', 'TF_repressor_top'] < 0
    assert scores_df.loc['cell_0', 'TF_activator_bottom'] < 0


# ------------------- SECTION 4: ROBUSTNESS & EDGE CASES -------------------

def test_no_side_effects_on_input_data(test_data):
    """
    What it tests: The function's safety and integrity.
    What it should do: NOT modify the user's original input AnnData or network
                      DataFrame. This prevents unexpected behavior in larger pipelines.
    """
    adata, net = test_data
    adata_original = adata.copy()
    net_original = net.copy()

    _func_kale(adata=adata, net=net, method='ranks_from_zscore')

    np.testing.assert_array_equal(adata_original.X, adata.X)
    pd.testing.assert_frame_equal(net_original, net)


def test_no_overlapping_targets(test_data):
    """
    What it tests: How the function handles a regulator whose targets do not
                   exist in the expression data.
    What it should do: Not crash. It should return NaN for the score and p-value
                      of that specific regulator, as no calculation is possible.
    """
    adata, _ = test_data
    net = pd.DataFrame({'source': ['TF_no_targets'], 'target': ['not_a_gene'], 'weight': [1]})
    scores, pvals = _func_kale(adata=adata, net=net, method='ranks_from_zscore')
    assert np.isnan(scores[0, 0]) and np.isnan(pvals[0, 0])


def test_all_nan_cell_input(test_data):
    """
    What it tests: Resilience to corrupted or missing data for an entire cell.
    What it should do: Produce an entire row of NaNs for the cell that has no
                      valid expression data, without affecting other cells.
    """
    adata, net = test_data
    adata.X[0, :] = np.nan
    scores, pvals = _func_kale(adata=adata, net=net, method='ranks_of_ranks')
    assert np.all(np.isnan(scores[0, :])) and np.all(np.isnan(pvals[0, :]))


def test_single_target_regulator():
    """
    What it tests: Stability of calculations when k=1 (a regulator has one target).
    What it should do: Run without math errors (e.g., division by zero) and
                      produce a finite, logical score.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10., 1.]}, index=['g1', 'g2']).T)
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g1'], 'weight': [1]})
    scores, pvals = _func_kale(adata=adata, net=net, method='ranks_of_ranks')
    assert np.isfinite(scores[0, 0]) and np.isfinite(pvals[0, 0])
    assert scores[0, 0] > 0  # Activating the top gene should yield a positive score.


# ------------------- SECTION 5: PARAMETER-SPECIFIC TESTS -------------------

def test_kale_min_targets_parameter(test_data):
    """
    What it tests: The functionality of the `min_targets` filter.
    What it should do: Exclude regulators that do not meet the minimum target
                      count. The output columns for these regulators should be
                      entirely NaN.
    """
    adata, net = test_data
    # TF1 has 3 targets, TF2 has 3, TF3 has 2. min_targets=3 should filter TF3.
    scores, _ = _func_kale(adata=adata, net=net, method='ranks_from_zscore', min_targets=3)

    all_sources = net['source'].unique()
    scores_df = pd.DataFrame(scores, columns=all_sources)

    assert not np.all(np.isnan(scores_df['TF1']))  # Should have valid scores
    assert not np.all(np.isnan(scores_df['TF2']))  # Should have valid scores
    assert np.all(np.isnan(scores_df['TF3']))  # Should be filtered out


def test_kale_ignore_zeros_parameter():
    """
    What it tests: The effect of the `ignore_zeros` flag.
    What it should do: Produce different numerical results when run with and
                      without the flag on a dataset containing zeros, because
                      the ranking universe (n_genes) changes.
    """
    adata = ad.AnnData(pd.DataFrame({'cell_0': [10, 5, 0]}, index=['g1', 'g2', 'g3']).T)
    net = pd.DataFrame({'source': ['TF1'], 'target': ['g1'], 'weight': [1]})

    scores_ignore, _ = _func_kale(adata=adata, net=net, method='ranks_of_ranks', ignore_zeros=True)
    scores_no_ignore, _ = _func_kale(adata=adata, net=net, method='ranks_of_ranks', ignore_zeros=False)

    assert not np.allclose(scores_ignore, scores_no_ignore)
