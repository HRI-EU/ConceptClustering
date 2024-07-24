# Concept Clustering package

## Installation
- install via pip: `pip install .`

## Usage
Checkout the example scripts in `./examples/`.
Basic usage:
```python
from concept_clustering.concept_clustering import ConceptClustering

ConceptClustering(
    description_spaces,
    n_clusters,
    max_iter,
).fit(X, centers)
```