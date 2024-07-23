"""
-*- coding: utf-8 -*-

 Copyright (C)
 Honda Research Institute Europe GmbH
 Carl-Legien-Str. 30
 63073 Offenbach/Main
 Germany
 UNPUBLISHED PROPRIETARY MATERIAL.
 ALL RIGHTS RESERVED.

 @author: Felix Lanfermann

concept clustering test energy management data set
"""

import numpy as np
import pandas as pd
from concept_clustering.concept_clustering import ConceptClustering

import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sns
from pathlib import Path

def create_centers(center_array, features_per_space):
    list_of_features = [item for sublist in features_per_space for item in sublist]
    df_center = pd.DataFrame(center_array, columns=list_of_features)
    return df_center


def plot_convergence(centers):
    num_clusters = len(centers[0])
    fig = plt.figure(figsize=(12, 12))

    for nf in range(len(centers[0][0])):
        df = pd.DataFrame()

        for nc in range(num_clusters):
            li = []
            for step in range(len(centers)):
                li.append(centers[step][nc, nf])
            df[f"cluster {nc}"] = li

            ax = plt.subplot2grid((len(centers[0][0]), num_clusters), (nf, nc), 1, 1)
            df.plot(
                y=f"cluster {nc}",
                ax=ax
            )

    plt.show()
    return None


# %% energy management data set
data = pd.read_csv("./energy.csv")
features_per_space = [
        ["Investment costs"],
        ["Yearly total costs", "posResilience"],
    ]
num_samples = data.shape[0]
num_clusters = 3
list_of_features = [item for sublist in features_per_space for item in sublist]
num_spaces = len(features_per_space)
df_centers = create_centers(
    np.array([
        [50000, 337000, 500],
        [200000, 333000, 1250],
        [500000, 326000, 3000]]),
    features_per_space=features_per_space
)

# kmeans_clustering = KMeans(n_clusters=2, random_state=0).fit(data)
# kmeans_labels = kmeans_clustering.labels_
# kmeans_centers = kmeans_clustering.cluster_centers_

ConClus = ConceptClustering(
    description_spaces=features_per_space,
    n_clusters=3,
    max_iter=50,
).fit(X=data, centers=df_centers)

all_centers = ConClus.all_centers_
all_labels = ConClus.all_labels_
all_concepts = ConClus.all_concepts_
centers = ConClus.cluster_centers_
labels = ConClus.labels_
concepts = ConClus.concepts_

for ns in range(num_spaces):
    data[f"labels_space{ns+1}"] = labels[ns]
data["concepts"] = concepts

df_concept_change = pd.DataFrame()
for step in range(len(all_concepts)):
    df_concept_change[f"{step}"] = all_concepts[step][:, 0]

dataset_concepts = data[data["concepts"].isin(range(num_clusters))]

# %% plot
fig = plt.figure(figsize=(12, 4))
ax1 = plt.subplot2grid((1, 2), (0, 1), 1, 1, fig=fig)
sns.scatterplot(
    data=data,
    x="Yearly total costs",
    y="posResilience",
    ax=ax1,
    color=[0.5, 0.5, 0.5, 0.3],
)
sns.scatterplot(
    data=dataset_concepts,
    hue="concepts",
    x="Yearly total costs",
    y="posResilience",
    palette="viridis_r",
    ax=ax1,
)

for i_f, fea in enumerate(["Investment costs"]):
    # get bins
    num_bins = 25
    bin_range_concepts = [(min(data[data["concepts"] == i_con][fea]),
                           max(data[data["concepts"] == i_con][fea])) for i_con in
                          range(num_clusters)]
    bins = np.sort(np.append(np.linspace(min(data[fea]), max(data[fea]), num_bins),
                             bin_range_concepts))

    ax2 = plt.subplot2grid((1, 2), (0, i_f), 1, 1, fig=fig)
    sns.histplot(
        data=data,
        x=fea,
        color=[0.9, 0.9, 0.9, 0.9],
        stat="count",
        bins=bins,
        element="step",
        log_scale=(False, False),
        ax=ax2
    )
    sns.histplot(
        data=dataset_concepts,
        x=fea,
        hue="concepts",
        palette="viridis_r",
        stat="count",
        bins=bins,
        element="step",
        log_scale=(False, False),
    )

path = Path(f"./example_results/")
path.mkdir(parents=True, exist_ok=True)

now = datetime.datetime.now().strftime('%Y-%m-%d')
fig.savefig(f"./example_results/{now}_concept_clustering_test_energy.png")

plt.tight_layout()
plt.show()

plot_convergence(all_centers)