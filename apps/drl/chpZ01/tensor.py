#
import numpy as np

class Tensor(object):
    def __init__(self, data, autograd=False, creators=None, creation_op=None, cid=None):
        self.data = data
        self.creators = creators
        self.creation_op = creation_op
        self.autograd = autograd
        self.grad = None
        self.children = {}
        if (cid is None):
            cid = np.random.randint(0, 10000000)
        self.cid = cid
        if creators is not None:
            for c in creators:
                if self.cid not in c.children:
                    c.children[self.cid] = 1
                else:
                    c.children[self.cid] += 1

    def all_children_grads_accounted_for(self):
        for cid, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                if self.children[grad_origin.cid] == 0:
                    raise Exception('cannot backprop more than once')
                else:
                    self.children[grad_origin.cid] -= 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                elif self.creation_op == 'neg':
                    self.creators[0].backward(self.grad.__neg__())
                elif self.creation_op == 'sub':
                    org = Tensor(self.grad.data)
                    self.creators[0].backward(org, self)
                    org = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(org, self)
                elif self.creation_op == 'mul':
                    rst = self.grad * self.creators[1]
                    self.creators[0].backward(rst, self)
                    rst = self.grad * self.creators[0]
                    self.creators[1].backward(rst, self)
                elif 'sum' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))
                elif 'expand' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    self.creators[0].backward(self.grad.sum(dim))


    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, creators=[self, other], creation_op='add')
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True, creators=[self], creation_op='neg')
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd:
            return Tensor(self.data - other.data, autograd=True, creators=[self, other], creation_op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, autograd=True, creators=[self, other], creation_op='mul')
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, creators=[self], creation_op='sum_' + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        print('trans_cmd1: {0};'.format(trans_cmd))
        trans_cmd.insert(dim, len(self.data.shape))
        print('trans_cmd2: {0};'.format(trans_cmd))
        new_shape = list(self.data.shape) + [copies]
        print('new_shape: {0};'.format(new_shape))
        new_data = self.data.repeat(copies).reshape(new_shape)
        print('1 new_data: {0}; {1};'.format(new_data.shape, new_data))
        new_data = new_data.transpose(trans_cmd)
        print('2 new_data: {0}; {1};'.format(new_data.shape, new_data))
        if self.autograd:
            return Tensor(new_data, autograd=True, creators=[self], creation_op='expand_'+str(dim))
        return Tensor(new_data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

    def to_string(self):
        ts = 'tensor_{0}:\r\n'.format(self.cid)
        ts = '{0}    data:{1};\r\n'.format(ts, self.data)
        ts = '{0}    autograd: {1};\r\n'.format(ts, self.autograd)
        ts = '{0}    creators: {1};\r\n'.format(ts, self.creators)
        ts = '{0}    creation_op: {1};\r\n'.format(ts, self.creation_op)
        ts = '{0}    cid: {1};\r\n'.format(ts, self.cid)
        ts = '{0}    grad: {1};\r\n'.format(ts, self.grad)
        ts = '{0}    children: {1};\r\n'.format(ts, self.children)
        return ts