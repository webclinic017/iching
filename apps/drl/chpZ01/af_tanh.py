# 双曲正切激活函数
from apps.drl.chpZ01.layer import Layer

class AfTanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()