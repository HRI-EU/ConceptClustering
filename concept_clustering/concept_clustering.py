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

concept clustering package
"""

import numpy as np
import pandas as pd


class ConceptClustering:
    """Concept Clustering

    Parameters
    ----------
    description_spaces :
        The spaces that the data set is split into for the identification of concepts

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    verbose : int, default=0
        Verbosity mode.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point for all spaces

    concepts_ :
        Concept association for each point

    all_centers_ :
        All centers from all steps

    all_labels_ :
        All labels from all steps for all spaces

    all_concepts_ :
        All concepts from all steps
    """

    def __init__(
        self,
        description_spaces,
        n_clusters=3,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
    ):

        self.description_spaces = description_spaces
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.all_centers_ = None
        self.all_labels_ = None
        self.all_concepts_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.concepts_ = None

    def _check_params(self, X):
        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if self.n_clusters <= 1:
            raise ValueError(f"n_clusters={self.n_clusters} should be >= 1.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= " f"n_clusters={self.n_clusters}."
            )

        # X is df
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X should be a pandas DataFrame, got {type(X)}")

    def fit(
        self, X, centers,
    ):
        """Compute concept clustering

        Parameters
        ----------
        X : pd.DataFrame

        centers : pd.DataFrame
        """
        self._check_params(X)

        data = X
        df_centers = centers
        num_samples = data.shape[0]
        num_clusters = self.n_clusters
        features_per_space = self.description_spaces
        num_spaces = len(features_per_space)
        iterations = self.max_iter

        # initialize all_centers
        all_centers = [np.array(df_centers)]
        all_labels = []
        all_concepts = []

        # loop
        for it in range(iterations):

            # initialize concepts
            concepts = -1 * np.ones((num_samples, 1))

            # calculate distances
            distances = []
            labels = []
            for ns in range(num_spaces):
                distances_ns = np.zeros((num_samples, num_clusters))
                for nc in range(num_clusters):
                    distances_ns_nc_a = np.linalg.norm(
                        data[features_per_space[ns]]
                        - df_centers[features_per_space[ns]].iloc[nc],
                        axis=1,
                    )
                    distances_ns[:, nc] = distances_ns_nc_a

                labels_space_ns = np.argmin(distances_ns, axis=1)
                labels.append(labels_space_ns)
                all_labels.append(labels)

            # update centers
            for nc in range(num_clusters):
                concept_nc = [
                    np.all([labels[ns] == nc for ns in range(num_spaces)], axis=0)
                ]
                concepts[concept_nc[0]] = nc
                for ns in range(num_spaces):
                    mean_alt_a = np.mean(
                        data[features_per_space[ns]][concept_nc[0]], axis=0
                    )
                    # if np.isnan(mean_alt_a[features_per_space[ns]]):
                    #     new_center_ns = centers[nc][ns]
                    # else:
                    new_center_ns = mean_alt_a[features_per_space[ns]]
                    df_centers.loc[nc, features_per_space[ns]] = new_center_ns

            all_concepts.append(concepts)
            all_centers.append(np.array(df_centers))

        self.all_centers_ = all_centers
        self.all_labels_ = all_labels
        self.all_concepts_ = all_concepts
        self.cluster_centers_ = all_centers[-1]
        self.labels_ = all_labels[-1]
        self.concepts_ = all_concepts[-1]
        return self
