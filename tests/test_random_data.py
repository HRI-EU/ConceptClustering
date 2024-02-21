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

concept clustering test 3
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


def create_random_data(features_per_space=[["feature 1"], ["feature 2"]], num_samples=100):
    num_spaces = len(features_per_space)
    list_of_features = [item for sublist in features_per_space for item in sublist]
    data_l = np.random.randn(num_samples, len(list_of_features))
    data = pd.DataFrame(data_l, columns=list_of_features)
    return data, list_of_features, num_spaces


def create_blob_data(features_per_space=[["feature 1"], ["feature 2"]], num_blobs=3, num_samples=100):
    num_spaces = len(features_per_space)
    list_of_features = [item for sublist in features_per_space for item in sublist]
    data_l = np.random.randn(num_samples, len(list_of_features))
    for i in range(num_blobs-1):
        data_l = np.append(data_l, np.random.randn(num_samples, len(list_of_features)) + np.array(len(list_of_features)*[(1+i)*3]), axis=0)
    data = pd.DataFrame(data_l, columns=list_of_features)
    return data, list_of_features, num_spaces


def create_centers(center_array, features_per_space):
    list_of_features = [item for sublist in features_per_space for item in sublist]
    df_center = pd.DataFrame(center_array, columns=list_of_features)
    return df_center


def make_palette(n_total_colors=10, c_start=(0, 0, 1), c_end=(0.5, 0.5, 0.5), c_special=(0.13, 0.56, 0.55)):
    """
    creates a color palette and cmap with similar colors and one special color
    :param n_total_colors: number colors for the palette {int}
    :param c_start: color 1 for gradient {1x3 array or list}
    :param c_end: color 2 for gradient {1x3 array or list}
    :param c_special: special color {1x3 array or list}
    :return: palette, cmap
    """
    # colormap with similar colors and one different
    vals = np.ones((n_total_colors, 4))
    vals[0:-1, 0] = np.linspace(c_start[0], c_end[0], n_total_colors - 1)
    vals[0:-1, 1] = np.linspace(c_start[1], c_end[1], n_total_colors - 1)
    vals[0:-1, 2] = np.linspace(c_start[2], c_end[2], n_total_colors - 1)
    vals[-1, 0:3] = c_special
    map = ListedColormap(vals)
    return vals, map


palette1 = sns.cubehelix_palette(n_colors=6, start=0.0, rot=-0.4, gamma=1.0, hue=0.99, light=0.75, dark=0.25,
                                 reverse=False, as_cmap=True)
palette2 = sns.cubehelix_palette(n_colors=6, start=1.3, rot=-0.4, gamma=1.0, hue=0.99, light=0.75, dark=0.25,
                                 reverse=False, as_cmap=True)

#%% plot 2 dim dataset
features_per_space = [["feature 1"], ["feature 2"]]
num_samples = 500
num_clusters = 2
data, list_of_features, num_spaces = create_random_data(features_per_space=features_per_space, num_samples=num_samples)
df_centers = create_centers(np.array([[1, 0.5], [0, 0]]), features_per_space=features_per_space)

ConClus = ConceptClustering(
    description_spaces=features_per_space,
    n_clusters=num_clusters,
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

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot2grid((1, 1), (0, 0), 1, 1, fig=fig)
sns.scatterplot(data=data, x="feature 1", y="feature 2", hue="concepts", ax=ax1)
for i_cen, cen in enumerate(all_centers):
    # for i_clu in range(num_clusters):
    sns.scatterplot(x=[cen[0][0]], y=[cen[0][1]], ax=ax1,
                    color=palette2(int(255 * (i_cen / (len(all_centers) - 1)))), s=100, marker="o")
    sns.scatterplot(x=[cen[1][0]], y=[cen[1][1]], ax=ax1,
                    color=palette1(int(255 * (i_cen / (len(all_centers) - 1)))), s=100, marker="o")
for i_cen, cen in enumerate(all_centers[1::]):
    sns.lineplot(x=[all_centers[i_cen][0][0], all_centers[i_cen+1][0][0]],
                 y=[all_centers[i_cen][0][1], all_centers[i_cen+1][0][1]],
                 color=palette2(int(255 * (i_cen / (len(all_centers) - 2)))))
    sns.lineplot(x=[all_centers[i_cen][1][0], all_centers[i_cen+1][1][0]],
                 y=[all_centers[i_cen][1][1], all_centers[i_cen+1][1][1]],
                 color=palette1(int(255*(i_cen/(len(all_centers)-2)))))


plt.tight_layout()
plt.show()

now = datetime.datetime.now().strftime('%Y-%m-%d')
fig.savefig(f"./test_results/{now}_concept_clustering_test_2dim.png")


# %% plot 3 dim dataset
features_per_space = [["feature 1"], ["feature 2", "feature 3"]]
num_samples = 500
num_clusters = 2
data, list_of_features, num_spaces = create_random_data(features_per_space=features_per_space, num_samples=num_samples)
df_centers = create_centers(np.array([[1, 0.5, 0.5], [0, 0, 0]]), features_per_space=features_per_space)

ConClus = ConceptClustering(
    description_spaces=features_per_space,
    n_clusters=num_clusters,
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

fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid((1, 2), (0, 0), 1, 1, fig=fig)
sns.scatterplot(data=data, x="feature 1", y="feature 1", hue="concepts", ax=ax1)
ax2 = plt.subplot2grid((1, 2), (0, 1), 1, 1, fig=fig)
sns.scatterplot(data=data, x="feature 2", y="feature 3", hue="concepts", ax=ax2)

for i_cen, cen in enumerate(all_centers):
    sns.scatterplot(x=[cen[0][1]], y=[cen[0][2]], ax=ax2,
                    color=palette2(int(255 * (i_cen / (len(all_centers) - 1)))), s=100, marker="o")
    sns.scatterplot(x=[cen[1][1]], y=[cen[1][2]], ax=ax2,
                    color=palette1(int(255 * (i_cen / (len(all_centers) - 1)))), s=100, marker="o")
for i_cen, cen in enumerate(all_centers[1::]):
    sns.lineplot(x=[all_centers[i_cen][0][1], all_centers[i_cen + 1][0][1]],
                 y=[all_centers[i_cen][0][2], all_centers[i_cen + 1][0][2]],
                 color=palette2(int(255 * (i_cen / (len(all_centers) - 2)))))
    sns.lineplot(x=[all_centers[i_cen][1][1], all_centers[i_cen + 1][1][1]],
                 y=[all_centers[i_cen][1][2], all_centers[i_cen + 1][1][2]],
                 color=palette1(int(255 * (i_cen / (len(all_centers) - 2)))))

now = datetime.datetime.now().strftime('%Y-%m-%d')
fig.savefig(f"./test_results/{now}_concept_clustering_test_3dim.png")

plt.tight_layout()
plt.show()

# improvement over steps
fig = plt.figure(figsize=(12, 8))

for i_f, feat in enumerate(list_of_features):
    ax1 = plt.subplot2grid((3, 1), (i_f, 0), 1, 1, fig=fig)
    for i_cen, cen in enumerate(all_centers):
        sns.scatterplot(x=range(len(all_centers))[i_cen], y=[cen[0][i_f]],
                        color=palette2(int(255 * (i_cen / (len(all_centers) - 1)))),
                        s=100, marker="o", ax=ax1)
        sns.scatterplot(x=range(len(all_centers))[i_cen], y=[cen[1][i_f]],
                        color=palette1(int(255 * (i_cen / (len(all_centers) - 1)))),
                        s=100, marker="o", ax=ax1)

plt.show()


# %% 4 dim example
features_per_space = [["feature a", "feature b"], ["feature c", "feature d"]]
num_samples = 300
num_clusters = 3
data, list_of_features, num_spaces = create_blob_data(features_per_space=features_per_space, num_samples=100)
df_centers = create_centers(np.array([[1, 0.5, 0.5, 0.7], [0, 0, 0, 0], [1, 1, 1, 1]]), features_per_space=features_per_space)

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

# plot 4 dim dataset

fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot2grid((1, 2), (0, 0), 1, 1, fig=fig)
sns.scatterplot(data=data, x="feature a", y="feature b", hue="concepts", ax=ax1, palette="viridis_r")

ax2 = plt.subplot2grid((1, 2), (0, 1), 1, 1, fig=fig)
sns.scatterplot(data=data, x="feature c", y="feature d", hue="concepts", ax=ax2, palette="viridis_r")


now = datetime.datetime.now().strftime('%Y-%m-%d')
fig.savefig(f"./test_results/{now}_concept_clustering_test_4dim.png")

plt.tight_layout()
plt.show()
