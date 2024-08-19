# Concept Clustering package

The concept clustering method separates

## Installation
- install via pip: `pip install .`

## Usage
Checkout the example scripts in `./examples/`.

Basic usage:
```python
import pandas as pd
from concept_clustering.concept_clustering import ConceptClustering

# define description spaces
features_per_space = [
        ["Investment costs"],
        ["Yearly total costs", "posResilience"],
    ]
list_of_features = [item for sublist in features_per_space for item in sublist]
num_clusters = 3  # set number of clusters

data = pd.read_csv("./energy.csv", usecols=list_of_features)
center_idx = [1, 4, 3]  # define initial centers
centers = data.loc[center_idx, list_of_features]

ConClus = ConceptClustering(
    description_spaces=features_per_space,
    n_clusters=num_clusters,
    max_iter=100,
).fit(X=data, centers=centers)

print(ConClus.concepts_)
```