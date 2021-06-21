#!/usr/bin/env python3

r"""
Standardized implementations of few-shot learning algorithms,
compatible with PyTorch Lightning.
"""

from learn2learn.algorithms.lightning_episodic_module import LightningEpisodicModule
from learn2learn.algorithms.lightning_maml import LightningMAML
from learn2learn.algorithms.lightning_anil import LightningANIL
from learn2learn.algorithms.lightning_protonet import LightningPrototypicalNetworks
from learn2learn.algorithms.lightning_metaoptnet import LightningMetaOptNet
