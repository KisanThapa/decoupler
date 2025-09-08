# KALE Method Implementation Guide

## Overview

This guide explains how to implement your **KALE** (Cell Activity Inference Landscape) method in the decoupler package. The method is currently integrated as a placeholder and ready for your implementation.

## What's Already Done

✅ **Method Registration**: KALE is registered in the decoupler methods system
✅ **Import Structure**: Method can be imported as `dc.mt.kale`
✅ **Testing Framework**: Tests are set up and passing
✅ **Documentation Structure**: Documentation files are created
✅ **Integration**: Method appears in `dc.mt.show()`

## What You Need to Implement

### 1. Core Algorithm (`src/decoupler/mt/_kale.py`)

**File**: `src/decoupler/mt/_kale.py`
**Function**: `_func_kale()`

**Current Status**: Placeholder that returns all zeros
**Your Task**: Replace the placeholder with your actual algorithm

#### Function Signature

```python
def _func_kale(
    mat: np.ndarray,        # Expression matrix (observations × features)
    adj: np.ndarray,        # Adjacency matrix (features × sources)
    verbose: bool = False,  # Verbosity flag
) -> tuple[np.ndarray, None]:
```

#### Input Parameters

-   `mat`: Expression matrix where rows are observations (cells/samples) and columns are features (genes)
-   `adj`: Adjacency matrix where rows are features and columns are sources (pathways/TFs)
-   `verbose`: Whether to print progress messages

#### Expected Output

-   `es`: Enrichment scores matrix (observations × sources)
-   `pv`: P-values (set to `None` if not applicable)

#### Implementation Steps

1. **Remove the placeholder code** (the loop that sets everything to 0.0)
2. **Implement your algorithm** using the input matrices
3. **Return the results** in the correct format
4. **Add progress tracking** if needed (use the `verbose` parameter)

### 2. Update Documentation (`docs/kale_method.md`)

**File**: `docs/kale_method.md`
**Your Task**: Replace placeholder descriptions with your actual method details

#### Sections to Update

-   **Method Description**: Explain how your algorithm works
-   **Mathematical Formulation**: Provide the actual equations
-   **Advantages**: List the benefits of your approach
-   **Use Cases**: Describe specific applications
-   **Technical Notes**: Include implementation details

### 3. Adjust Tests (if needed) (`tests/mt/test_kale.py`)

**File**: `tests/mt/test_kale.py`
**Your Task**: Update tests if your method has different expected behavior

#### Current Test Assumptions

-   Method returns finite scores
-   Scores are within reasonable bounds (-10 to +10)
-   Method handles edge cases gracefully
-   Method works with different weight configurations

#### If Your Method Differs

-   Update score range expectations
-   Modify edge case handling
-   Adjust weight handling tests

## Implementation Example

Here's a template for your implementation:

```python
def _func_kale(
    mat: np.ndarray,
    adj: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, None]:
    """
    KALE (Cell Activity Inference Landscape) method.

    Your method description here...
    """
    nobs, nvar = mat.shape
    nsrc = adj.shape[1]

    m = f"kale - calculating {nsrc} scores across {nobs} observations"
    _log(m, level="info", verbose=verbose)

    # Initialize results matrix
    es = np.zeros((nobs, nsrc))

    # TODO: Implement your algorithm here
    #
    # Example structure:
    # for i in tqdm(range(nobs), disable=not verbose):
    #     for j in range(nsrc):
    #         # Your calculation here
    #         es[i, j] = your_algorithm(mat[i, :], adj[:, j])

    return es, None
```

## Testing Your Implementation

### 1. Run the Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run KALE tests
python -m pytest tests/mt/test_kale.py -v

# Run all method tests
python -m pytest tests/mt/test_methods.py -v
```

### 2. Test with Example Data

```bash
# Run the example script
python examples/kale_example.py

# Test in Python
python -c "
import decoupler as dc
adata, net = dc.ds.toy()
dc.mt.kale(adata, net, tmin=3)
print('Scores shape:', adata.obsm['score_kale'].shape)
print('Sample scores:', adata.obsm['score_kale'].iloc[:3, :3])
"
```

### 3. Verify Integration

```bash
# Check that KALE appears in methods
python -c "
import decoupler as dc
print(dc.mt.show())
"
```

## Common Issues and Solutions

### Issue: Method Not Found

**Solution**: Make sure you've imported and registered the method in `_methods.py`

### Issue: Tests Failing

**Solution**: Check that your output matches the expected format and ranges

### Issue: Import Errors

**Solution**: Ensure all dependencies are properly imported

### Issue: Wrong Output Shape

**Solution**: Verify that `es.shape == (nobs, nsrc)`

## Next Steps

1. **Implement your algorithm** in `_func_kale()`
2. **Update the documentation** in `docs/kale_method.md`
3. **Test your implementation** thoroughly
4. **Update tests** if needed
5. **Verify integration** works correctly

## Getting Help

-   Check existing method implementations in `src/decoupler/mt/` for examples
-   Review the decoupler documentation and API
-   Test with small datasets first
-   Use the verbose flag for debugging

---

**Good luck with your implementation!** 🚀

Once you're done, your KALE method will be fully functional and available to all decoupler users.
