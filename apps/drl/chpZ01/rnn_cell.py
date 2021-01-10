#
import numpy as np
from apps.drl.chpZ01.tensor import Tensor
from apps.drl.chpZ01.layer import Layer
from apps.drl.chpZ01.linear import Linear
from apps.drl.chpZ01.af_tanh import AfTanh
from apps.drl.chpZ01.af_sigmoid import AfSigmoid

class RnnCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        if(activation == 'sigmoid'):
            self.activation = AfSigmoid()
        elif(activation == 'tanh'):
            self.activation == AfTanh()
        else:
            raise Exception("Non-linearity not found")
        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)
        self.parameters = []
        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()        
    
    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden
    
    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)