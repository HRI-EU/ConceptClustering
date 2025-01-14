"""
-*- coding: utf-8 -*-

concept clustering package

Copyright (C)
Honda Research Institute Europe GmbH
Carl-Legien-Str. 30
63073 Offenbach/Main
Germany

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


class ConceptClustering:
    """Concept Clustering

    Parameters
    ----------
    description_spaces : list of strings
        The spaces that the data set is split into for the identification of concepts

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default=300
        Maximum number of iterations of the algorithm for a
        single run.

    verbose : int, default=0
        Verbosity mode. Not yet implemented.

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

    def _check_params(self, X, centers):
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
            raise TypeError(f"X should be a pandas DataFrame, got {type(X)}.")

        # X is numeric
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise TypeError(
                f"X should only contain numeric values. Consider encoding X."
            )

        # centers is df
        if not isinstance(centers, pd.DataFrame):
            raise TypeError(
                f"centers should be a pandas DataFrame, got {type(centers)}."
            )

        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"Number of centers and number of clusters are not equal: {centers.shape[0]} != {self.n_clusters}"
            )

    def fit(
        self, X, centers,
    ):
        """Compute concept clustering

        Parameters
        ----------
        X : pd.DataFrame

        centers : pd.DataFrame
        """
        self._check_params(X, centers)

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

    def concept_quality(self, num_c=None, labels=None, concepts=None):
        """Compute concept quality

        Parameters
        ----------
        num_c : int
        labels : ndarray of shape (n_samples, num_ds)
        concepts : ndarray of shape (n_samples, 1)

        Returns
        -------
        cqm: list
            concept quality metric (separately for each concept)

        """
        if num_c is None:
            num_c = self.n_clusters
        if labels is None:
            labels = self.labels_
        if concepts is None:
            concepts = self.concepts_
        num_ds = len(labels)

        # Create DataFrame directly with labels and concepts
        df = pd.DataFrame({f"labels_{i}": labels[i] for i in range(num_ds)})
        df["concepts"] = concepts.astype(int)

        # Precompute value counts for concepts and labels
        concept_counts = df["concepts"].value_counts()
        label_counts = {
            f"labels_{i}": df[f"labels_{i}"].value_counts() for i in range(num_ds)
        }

        # Calculate CQM using vectorized operations
        cqm = []
        for n_c in range(num_c):
            if n_c not in concept_counts.index:
                cqm.append(0)
            else:
                ratios = np.array(
                    [
                        concept_counts[n_c] / label_counts[f"labels_{n_s}"][n_c]
                        for n_s in range(num_ds)
                    ]
                )
                cqm_value = np.prod(ratios) ** (1 / num_ds)
                cqm.append(cqm_value)

        return cqm
