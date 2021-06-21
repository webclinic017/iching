#!/usr/bin/env python3

r"""
A set of high-level algorithm implementations, with easy-to-use API.
"""

from learn2learn.algorithms.maml import MAML, maml_update
from learn2learn.algorithms.meta_sgd import MetaSGD, meta_sgd_update
from learn2learn.algorithms.gbml import GBML
from learn2learn.algorithms.lightning import (
    LightningEpisodicModule,
    LightningMAML,
    LightningANIL,
    LightningPrototypicalNetworks,
    LightningMetaOptNet,
)
