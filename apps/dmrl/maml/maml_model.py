# Import modules we need
import torch
import torch.nn as nn
import torch.nn.functional as F

def LinearBlock(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.BatchNorm1d(out_size),
        nn.ReLU()
    )

def LinearBlockFunction(x, w, b, w_bn, b_bn):
    x = F.linear(x, w, b)
    x = F.batch_norm(x, running_mean = None, running_var = None, 
                weight = w_bn, bias = b_bn, training = True)
    x = F.relu(x)
    return x

class MamlModel(nn.Module):
    def __init__(self, in_size, n_way):
        super(MamlModel, self).__init__()
        self.linear1 = LinearBlock(in_size, 64)
        self.linear2 = LinearBlock(64, 32)
        self.linear3 = LinearBlock(32, 16)
        self.logits = nn.Linear(16, n_way)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        #x = nn.Flatten(x)
        x = x.flatten(start_dim=1)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，
                以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        '''
        for block in [1, 2, 3]:
            x = LinearBlockFunction(x,
                params[f'linear{block}.0.weight'],
                params[f'linear{block}.0.bias'],
                params.get(f'linear{block}.1.weight'),
                params.get(f'linear{block}.1.bias')
            )
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'] , params['logits.bias'])
        return x