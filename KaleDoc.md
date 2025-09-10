# Using the KALE Method in Decoupler

**KALE (Cell Activity Inference Landscape)** is a flexible and robust method for inferring transcription factor (TF) activity from gene expression data. It is a rank-based approach that offers multiple statistical variants, including both unweighted and weighted models, to suit different types of prior knowledge networks.


```python
import decoupler as dc

# Pass the full AnnData object to the `mat` parameter
dc.mt.kale(
    adata=adata,
    net=net,
    method='ranks_from_zscore',
    # ... other KALE-specific arguments
)
```

## KALE-Specific Parameters

These parameters are passed directly to the `run_kale` function to control its behavior.

| Parameter      | Type | Description | Default |
|:---------------| :--- | :--- | :--- |
| `adata`        | `AnnData` | The input `AnnData` object containing the gene expression matrix in `.X`. This is the primary data source for KALE. | |
| `net`          | `DataFrame` | A network DataFrame with `source` (TF), `target` (gene), and `weight` columns. The sign of the `weight` indicates activation (+) or repression (-). | |
| `method`       | `str` | The specific statistical variant of KALE to use. This is the most important parameter. See the table below for options. | `'ranks_from_zscore'` |
| `min_targets`  | `int` | The minimum number of targets a TF must have in the network to be considered. TFs with fewer targets will have their scores returned as `NaN`. | `0` |
| `ignore_zeros` | `bool` | If `True`, zeros in the expression matrix are treated as missing data (`NaN`). This is highly recommended for single-cell data. | `True` |

### KALE Analysis Methods (`method` options)

| `method` String | Input Data | Weighting | Description |
| :--- | :--- | :--- | :--- |
| **`ranks_from_zscore`** | Gene Z-scores | Unweighted | **(Default)** Ranks genes based on their Z-scores across all cells. A robust, general-purpose choice. |
| **`weighted_ranks_from_zscore`** | Gene Z-scores | Weighted | Same as above, but uses the `weight` column from the network to give more influence to high-confidence interactions. |
| **`ranks_of_ranks`** | Raw Expression | Unweighted | Ranks genes directly from their expression values within each cell. Skips Z-score normalization. |
| **`weighted_ranks_of_ranks`** | Raw Expression | Weighted | Same as above, but incorporates network weights into the rank aggregation. |
| **`stouffers_zscore`** | Gene Z-scores | Unweighted | A different approach that combines the Z-scores of target genes using Stouffer's method instead of ranks. |



## Output

*   **TF Activity Score:** `adata.obsm['score_kale']` (a DataFrame with cells as rows and TFs as columns).
*   **P-values:** `adata.obsm['kale_pvals']` (a DataFrame with the same shape).
