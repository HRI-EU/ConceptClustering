#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  launches the unit testing
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

# ----------------------------------------------------------------------------
# Includes
# ----------------------------------------------------------------------------
import logging
import unittest

from concept_clustering.concept_clustering import ConceptClustering


# ----------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------
class TestConceptClustering(unittest.TestCase):
    def test_f1(self):
        a = 5
        assert a == 5


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # launch all functions named "test_..." in sequence
    unittest.main()


# EOF
