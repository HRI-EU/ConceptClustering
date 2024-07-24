#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  testing
#
#  Copyright (C)
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#

import sys
import pytest
import unittest

import numpy as np

from concept_clustering.concept_clustering import ConceptClustering


def test_f1():
    a = 5
    assert a == 5


def test_f2():
    b = 5
    assert b == 5


def test_f3():
    c = 5
    assert c == 5


if __name__ == "__main__":
    pytest.main()
