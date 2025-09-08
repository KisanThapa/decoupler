import numpy as np

import decoupler as dc


def test_func_kale(
    mat,
    adjmat,
):
    """Test the kale method function."""
    X, obs, var = mat
    es, pv = dc.mt._kale._func_kale(mat=X, adj=adjmat)

    # Check that enrichment scores are finite
    assert np.isfinite(es).all()

    # Check that p-values are None (as specified in the method)
    assert pv is None

    # Check output shape
    assert es.shape[0] == X.shape[0]  # Number of observations
    assert es.shape[1] == adjmat.shape[1]  # Number of feature sets

    # Check that scores are finite and within reasonable bounds
    # Scores can be negative due to negative weights, but should be finite
    assert np.isfinite(es).all()
    # Scores should be within reasonable bounds (not too extreme)
    assert (es >= -10).all()
    assert (es <= 10).all()


def test_kale_method(
    adata,
    net,
):
    """Test the kale method through the main interface."""
    # Test with default parameters
    result = dc.mt.kale(adata, net, tmin=3)

    # The method modifies the AnnData object in-place and returns None
    # Check that scores are added to obsm
    assert "score_kale" in adata.obsm

    # Check that scores have the right shape
    scores = adata.obsm["score_kale"]
    assert scores.shape[0] == adata.shape[0]  # Number of observations
    assert scores.shape[1] == net["source"].nunique()  # Number of sources

    # Check that scores are finite
    assert np.isfinite(scores.values).all()


def test_kale_edge_cases():
    """Test edge cases for the kale method."""
    # Test with single observation
    single_mat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    single_adj = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    es, pv = dc.mt._kale._func_kale(mat=single_mat, adj=single_adj)
    assert es.shape == (1, 3)
    assert pv is None

    # Test with all zero weights
    zero_mat = np.array([[1.0, 2.0, 3.0]])
    zero_adj = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    es, pv = dc.mt._kale._func_kale(mat=zero_mat, adj=zero_adj)
    assert es.shape == (1, 2)
    assert np.all(es == 0.0)  # All scores should be zero


def test_kale_weights():
    """Test that the kale method properly handles different weight configurations."""
    # Test matrix
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Test with different weight patterns
    # Case 1: All positive weights
    adj_pos = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.8]])

    es, pv = dc.mt._kale._func_kale(mat=mat, adj=adj_pos)
    assert es.shape == (2, 2)
    assert np.all(es >= 0)  # All scores should be positive with positive weights

    # Case 2: Mixed positive and negative weights
    adj_mixed = np.array([[1.0, -0.5], [0.5, 1.0], [-0.3, 0.8]])

    es, pv = dc.mt._kale._func_kale(mat=mat, adj=adj_mixed)
    assert es.shape == (2, 2)

    # Case 3: Zero weights (should result in zero scores)
    adj_zero = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    es, pv = dc.mt._kale._func_kale(mat=mat, adj=adj_zero)
    assert es.shape == (2, 2)
    assert np.all(es == 0.0)  # All scores should be zero
