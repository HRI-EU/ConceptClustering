#
#  Custom package settings
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


name = "concept_clustering"

version = "0.1"

category = "Libraries"

envVars = [("PYTHONPATH", "${INSTALL_ROOT}/include:${PYTHONPATH}")]

sqLevel = "basic"

sqOptOutRules = ["GEN03"]
