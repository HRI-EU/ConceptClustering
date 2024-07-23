#!/bin/bash
#
#  launches the unittest suite
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
#  Start by hand using "./unittest.sh", or programmatically via "BST.py -t"
#  (or BuildSystemTools.unittest() in Python).
#
#


#----------------------------------------------------------------------------
# Includes
#----------------------------------------------------------------------------


source ${TOOLBOSCORE_ROOT}/include/Unittest.bash


#----------------------------------------------------------------------------
# Launcher
#----------------------------------------------------------------------------


export PYTHONPATH=$(pwd)/src:${PYTHONPATH}


# list all unittests here, comment-out some to run particular test only
runTest ./test/Test_concept_clustering.py


# we managed to get here --> success
exit 0


# EOF
