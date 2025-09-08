# KALE Method Documentation

## Overview

**KALE** (Cell Activity Inference Landscape) is a method for computing enrichment scores to infer cell activity landscapes from gene expression data and regulatory networks.

## Method Description

The KALE method computes enrichment scores by:

1. **Cell Activity Inference**: Analyzes gene expression patterns to infer cellular activity states
2. **Landscape Mapping**: Maps the activity landscape across different cell populations or conditions
3. **Network Integration**: Integrates regulatory network information to enhance activity inference

_Note: This is a placeholder description. The actual implementation details will be provided by the method developer._

## Mathematical Formulation

_The mathematical formulation for the KALE method will be provided by the method developer._

_This section will contain the specific equations and algorithms used in the Cell Activity Inference Landscape approach._

## Usage

### Basic Usage

```python
import decoupler as dc

# Load data
adata, net = dc.ds.toy()

# Run KALE method
result = dc.mt.kale(adata, net, tmin=3)

# Access results
scores = adata.obsm['score_kale']
```

### Parameters

-   `data`: Input data (AnnData, pandas DataFrame, or numpy array)
-   `net`: Network with source-target-weight relationships
-   `tmin`: Minimum number of targets per source (default: 5)
-   `verbose`: Whether to print progress messages (default: False)

### Returns

The method modifies the input AnnData object in-place by adding:

-   `adata.obsm['score_kale']`: Enrichment scores matrix

## Method Characteristics

-   **Type**: Numerical
-   **Adjacency Matrix**: Required
-   **Weights**: Supported
-   **Statistical Testing**: Not available
-   **Score Limits**: \((-\infty, +\infty)\)

## Advantages

_The advantages and benefits of the KALE method will be described by the method developer._

_This section will highlight the specific strengths and unique features of the Cell Activity Inference Landscape approach._

## Use Cases

_The specific use cases and applications of the KALE method will be described by the method developer._

_This section will outline the types of analyses and research questions that the Cell Activity Inference Landscape method is designed to address._

## Example

```python
import decoupler as dc
import matplotlib.pyplot as plt

# Load data and run KALE
adata, net = dc.ds.toy()
dc.mt.kale(adata, net, tmin=3)

# Visualize results
scores = adata.obsm['score_kale']
plt.figure(figsize=(10, 6))
plt.imshow(scores.T, aspect='auto', cmap='RdBu_r', center=0)
plt.colorbar(label='Enrichment Score')
plt.xlabel('Observations')
plt.ylabel('Sources')
plt.title('KALE Enrichment Scores')
plt.show()
```

## References

This is a custom implementation developed for the decoupler package. The method combines concepts from kernel methods and network-based enrichment analysis.

## Technical Notes

_The technical implementation details and considerations for the KALE method will be provided by the method developer._

_This section will include information about algorithm complexity, memory usage, performance characteristics, and any specific technical requirements._
