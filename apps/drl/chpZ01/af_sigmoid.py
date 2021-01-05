#
from apps.drl.chpZ01.layer import Layer

class AfSigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()