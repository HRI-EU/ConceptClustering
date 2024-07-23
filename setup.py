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

Script for installing the Concept Clustering package.

## LICENSE: GPL 3.0
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages

setup(
    name='concept_clustering',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
    author='Felix Lanfermann',
    author_email='felix.lanfermann@honda-ri.de',
    description='Clustering data into concepts based on multiple description spaces',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='TBC',
    classifiers=[
        'Programming Language :: Python :: TBD',
        'License :: TBD',
        'Operating System :: TBD',
    ],
    python_requires='>=3.7',
)
