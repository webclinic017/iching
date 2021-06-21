#!/usr/bin/env python3

"""
Optimization transforms are special modules that take gradients as inputs
and output model updates.
Transforms are usually parameterized, and those parameters can be learned by
gradient descent, allow you to learn optimization functions from data.
"""

from learn2learn.module_transform import ModuleTransform, ReshapedTransform
from learn2learn.kronecker_transform import KroneckerTransform
from learn2learn.transform_dictionary import TransformDictionary
from learn2learn.metacurvature_transform import MetaCurvatureTransform
