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

concept clustering test 2
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

# %% airfoil example
data = pd.read_csv("./airfoils.csv")
features_per_space = [
    ["p0", "p1", "p2", "p3"],
    ["camber_0.03", "camber_0.2", "camber_0.4", "camber_0.6", "camber_0.8"],
    ["U10_a0_OF_cl", "U10_a0_OF_cd"],
    ["U10_a1_OF_cl", "U10_a1_OF_cd"],
    ["U10_a3_OF_cl", "U10_a3_OF_cd"]
]

list_of_features = [item for sublist in features_per_space for item in sublist]
archetypes = [11477, 5456, 9231]
arch_data = data[data["Index"].isin(archetypes)][list_of_features]
num_samples = data.shape[0]
num_clusters = 3
num_spaces = len(features_per_space)
df_centers = create_centers(
    np.array(arch_data),
    features_per_space=features_per_space
)

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

concepts = all_concepts[-1]
for ns in range(num_spaces):
    data[f"labels_space{ns+1}"] = labels[ns]
data["concepts"] = concepts

df_concept_change = pd.DataFrame()
for step in range(len(all_concepts)):
    df_concept_change[f"{step}"] = all_concepts[step][:, 0]

dataset_concepts = data[data["concepts"].isin(range(num_clusters))]

# %%
fig = plt.figure(figsize=(12, 5))
for i_ds, ds in enumerate(features_per_space):
    if len(ds) == 2:
        ax = plt.subplot2grid((1, len(features_per_space)), (0, i_ds), 1, 1)
        sns.scatterplot(
            data=data,
            x=ds[0],
            y=ds[1],
            ax=ax,
            color=[0.5, 0.5, 0.5, 0.3],
            s=5,
        )
        sns.scatterplot(
            data=dataset_concepts,
            hue="concepts",
            x=ds[0],
            y=ds[1],
            palette="viridis_r",
            ax=ax,
            s=5
        )
        ax.legend([], [], frameon=False)  # remove legend

    else:
        for sub_ds_i, sub_ds in enumerate(ds):

            # get bins
            num_bins = 25
            bin_range_concepts = [(min(data[data["concepts"] == i_con][sub_ds]),
                                   max(data[data["concepts"] == i_con][sub_ds])) for i_con in
                                  range(num_clusters)]
            bins = np.sort(np.append(np.linspace(min(data[sub_ds]), max(data[sub_ds]), num_bins),
                                     bin_range_concepts))

            # ax = plt.subplot2grid((len(ds), len(description_spaces)), (sub_ds_i, i_ds), 1, 1,)
            # tight layout only works for grid-like subplots
            row_number = max(len(desc) for desc in features_per_space)
            ax = plt.subplot2grid((row_number, len(features_per_space)), (sub_ds_i, i_ds), 1, 1,)
            sns.histplot(
                data=data,
                x=sub_ds,
                color=[0.9, 0.9, 0.9, 0.9],
                stat="count",
                bins=bins,
                element="step",
                log_scale=(False, False),
            )
            sns.histplot(
                data=dataset_concepts,
                x=sub_ds,
                hue="concepts",
                palette="viridis_r",
                stat="count",
                bins=bins,
                element="step",
                log_scale=(False, True),
            )
            handles, _ = ax.get_legend_handles_labels()
            ax.legend([], [], frameon=False)  # remove legend

fig.tight_layout(w_pad=2, h_pad=0)  # w_pad=3, h_pad=2
now = datetime.datetime.now().strftime('%Y-%m-%d')
fig.savefig(f"./test_results/{now}_concept_clustering_test_airfoil.png")
plt.show()

plot_convergence(all_centers)
