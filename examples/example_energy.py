"""
-*- coding: utf-8 -*-

energy management concept clustering example

Copyright (c) 2025, Honda Research Institute Europe GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 @author: Felix Lanfermann
"""

import numpy as np
import pandas as pd
from concept_clustering.concept_clustering import ConceptClustering

import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
            df.plot(y=f"cluster {nc}", ax=ax)

    plt.show()
    return None


# %% energy management data set
features_per_space = [
    ["Investment costs"],
    ["Yearly total costs", "posResilience"],
]
num_clusters = 3
use_scaled_data = False

list_of_features = [item for sublist in features_per_space for item in sublist]
num_spaces = len(features_per_space)
data = pd.read_csv("./energy.csv", usecols=list_of_features)
num_samples = data.shape[0]

# set the initial centers manually
df_centers = create_centers(
    np.array([[70000, 336000, 500], [250000, 333000, 1000], [450000, 324000, 1000]]),
    features_per_space=features_per_space,
)

if use_scaled_data:
    # scale the data to [0, 1]
    scaler = MinMaxScaler().fit(data[list_of_features])
    data_scaled = pd.DataFrame(
        scaler.transform(data[list_of_features]), columns=data[list_of_features].columns
    )

    df_centers_scaled = pd.DataFrame(
        scaler.transform(df_centers), columns=data[list_of_features].columns
    )

    ConClus = ConceptClustering(
        description_spaces=features_per_space, n_clusters=3, max_iter=50,
    ).fit(X=data_scaled, centers=df_centers_scaled)
else:
    ConClus = ConceptClustering(
        description_spaces=features_per_space, n_clusters=3, max_iter=50,
    ).fit(X=data, centers=df_centers)

all_centers = ConClus.all_centers_
all_labels = ConClus.all_labels_
all_concepts = ConClus.all_concepts_
centers = ConClus.cluster_centers_
labels = ConClus.labels_
concepts = ConClus.concepts_
cqm = ConClus.concept_quality()

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
    bin_range_concepts = [
        (
            min(data[data["concepts"] == i_con][fea]),
            max(data[data["concepts"] == i_con][fea]),
        )
        for i_con in range(num_clusters)
    ]
    bins = np.sort(
        np.append(
            np.linspace(min(data[fea]), max(data[fea]), num_bins), bin_range_concepts
        )
    )

    ax2 = plt.subplot2grid((1, 2), (0, i_f), 1, 1, fig=fig)
    sns.histplot(
        data=data,
        x=fea,
        color=[0.9, 0.9, 0.9, 0.9],
        stat="count",
        bins=bins,
        element="step",
        log_scale=(False, False),
        ax=ax2,
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

now = datetime.datetime.now().strftime("%Y-%m-%d")
fig.savefig(f"./example_results/{now}_concept_clustering_test_energy.png")

plt.tight_layout()
plt.show()

plot_convergence(all_centers)
