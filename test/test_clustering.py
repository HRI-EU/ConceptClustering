"""
-*- coding: utf-8 -*-

test file

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

import sys
import pytest
import unittest

import numpy as np
import pandas as pd

from concept_clustering.concept_clustering import ConceptClustering


def test_dummy():
    a = 5
    assert a == 5


def test_check_params_max_iter():
    with pytest.raises(ValueError):
        ConceptClustering(description_spaces=None, n_clusters=3, max_iter=-50,).fit(
            X=None, centers=None
        )


def test_check_params_n_clusters():
    with pytest.raises(ValueError):
        ConceptClustering(description_spaces=None, n_clusters=-3, max_iter=50,).fit(
            X=None, centers=None
        )


def test_check_params_n_clusters_x_shape():
    data = pd.DataFrame(np.array([[1, 2], [4, 5]]), columns=["a", "b"])
    with pytest.raises(ValueError):
        ConceptClustering(description_spaces=None, n_clusters=3, max_iter=50,).fit(
            X=data, centers=None
        )


def test_check_params_x_type():
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    with pytest.raises(TypeError):
        ConceptClustering(description_spaces=None, n_clusters=3, max_iter=50,).fit(
            X=data, centers=None
        )


def test_check_params_centers_type():
    data = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"])
    centers = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(TypeError):
        ConceptClustering(description_spaces=None, n_clusters=3, max_iter=50,).fit(
            X=data, centers=centers
        )


def test_check_params_centers_number():
    data = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"])
    centers = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b", "c"])
    with pytest.raises(ValueError):
        ConceptClustering(description_spaces=None, n_clusters=3, max_iter=50,).fit(
            X=data, centers=centers
        )


if __name__ == "__main__":
    pytest.main()
