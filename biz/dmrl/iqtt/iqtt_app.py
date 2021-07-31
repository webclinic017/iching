# 
import torch
from biz.dmrl.iqtt.self_attention import SelfAttention

class IqttApp(object):
    def __init__(self):
        self.name = 'biz.dmrl.iqtt.iqtt_app.IqttApp'

    def startup(self, args={}):
        print('Iching Quantitative Trading Transformer v0.0.1')
        sa = SelfAttention(emb=5, heads=2, mask=False)
        x = torch.rand(8, 10, 5)
        y = sa(x)
        print('x: {0}; y: {1};'.format(x.shape, y.shape))