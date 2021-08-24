#
import torch
from torch import nn
import torch.nn.functional as F
from biz.dmrl.iqtt.iqtt_config import IqttConfig
from biz.dmrl.iqtt.iqtt_util import IqttUtil
from biz.dmrl.iqtt.transformer_block import TransformerBlock

class IqttTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False, app_mode=1):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()
        if IqttConfig.APP_MODE_IMDB == app_mode:
            self.task_mode = IqttConfig.TASK_MODE_RANDOM
        else:
            self.task_mode = IqttConfig.TASK_MODE_TS
        self.app_mode = app_mode
        self.num_tokens, self.max_pool = num_tokens, max_pool
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, task_mode=self.task_mode))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        if IqttConfig.APP_MODE_IMDB == self.app_mode:
            tokens = self.token_embedding(x)
            b, t, e = tokens.size()
            positions = self.pos_embedding(torch.arange(t, device=IqttUtil.d()))[None, :, :].expand(b, t, e)
            x = tokens + positions
        x = self.do(x)
        x = self.tblocks(x)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)