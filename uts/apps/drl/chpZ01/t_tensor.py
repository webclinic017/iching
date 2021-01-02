#
import numpy as np
import unittest
from apps.drl.chpZ01.tensor import Tensor

class TTensor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_init_001(self):
        a = Tensor(np.array([1, 2, 3, 4, 5]), autograd=True)
        b = Tensor(np.array([10, 10, 10, 10, 10]), autograd=True)
        c = Tensor(np.array([5, 4, 3, 2, 1]), autograd=True)
        d = a + b
        e = b + c
        f = d + e
        print('f: {0};'.format(f.to_string()))
        print('d: {0};'.format(d.to_string()))
        print('e: {0};'.format(e.to_string()))
        print('a: {0};'.format(a.to_string()))
        print('b: {0};'.format(b.to_string()))
        print('c: {0};'.format(c.to_string()))

    def test_add_backward_001(self):
        a = Tensor(np.array([1, 2, 3, 4, 5]), autograd=True)
        b = Tensor(np.array([10, 10, 10, 10, 10]), autograd=True)
        c = Tensor(np.array([5, 4, 3, 2, 1]), autograd=True)
        d = a + b
        e = b + c
        f = d + e
        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print('f: {0};'.format(f.to_string()))
        print('d: {0};'.format(d.to_string()))
        print('e: {0};'.format(e.to_string()))
        print('a: {0};'.format(a.to_string()))
        print('b: {0};'.format(b.to_string()))
        print('c: {0};'.format(c.to_string()))

    def test_neg_backward_001(self):
        a = Tensor(np.array([1, 2, 3, 4, 5]), autograd=True)
        b = Tensor(np.array([10, 10, 10, 10, 10]), autograd=True)
        c = Tensor(np.array([5, 4, 3, 2, 1]), autograd=True)
        d = a + (-b)
        e = (-b) + c
        f = d + e
        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print('f: {0};'.format(f.to_string()))
        print('d: {0};'.format(d.to_string()))
        print('e: {0};'.format(e.to_string()))
        print('a: {0};'.format(a.to_string()))
        print('b: {0};'.format(b.to_string()))
        print('c: {0};'.format(c.to_string()))

    def test_sub_backward_001(self):
        a = Tensor(np.array([1, 2, 3, 4, 5]), autograd=True)
        b = Tensor(np.array([10, 10, 10, 10, 10]), autograd=True)
        c = Tensor(np.array([5, 4, 3, 2, 1]), autograd=True)
        d = a + b
        e = b - c
        f = d + e
        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print('f: {0};'.format(f.to_string()))
        print('d: {0};'.format(d.to_string()))
        print('e: {0};'.format(e.to_string()))
        print('a: {0};'.format(a.to_string()))
        print('b: {0};'.format(b.to_string()))
        print('c: {0};'.format(c.to_string()))

    def test_mul_backward_001(self):
        a = Tensor(np.array([1, 2, 3, 4, 5]), autograd=True)
        b = Tensor(np.array([10, 10, 10, 10, 10]), autograd=True)
        c = Tensor(np.array([5, 4, 3, 2, 1]), autograd=True)
        d = a + b
        e = b - c
        f = d * e
        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print('f: {0};'.format(f.to_string()))
        print('d: {0};'.format(d.to_string()))
        print('e: {0};'.format(e.to_string()))
        print('a: {0};'.format(a.to_string()))
        print('b: {0};'.format(b.to_string()))
        print('c: {0};'.format(c.to_string()))

    def test_sum_001(self):
        v = Tensor(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]))
        print(v.sum(0))
        print(v.sum(1))

    def test_expand_001(self):
        v = Tensor(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]))
        print('v.expand: {0}'.format(v.expand(0, 4)))

    def test_expand_002(self):
        v = Tensor(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]))
        print('v.expand: {0}'.format(v.expand(1, 4)))

    def test_sum_grad_001(self):
        v = Tensor(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]), autograd=True)
        u = v.sum(0)
        u.backward(Tensor(np.array([1, 1, 1])))
        print('grad: {0};'.format(v.to_string()))

    def test_transpose_001(self):
        v = Tensor(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]), autograd=True)
        v_t = v.transpose()
        print('v_t: \r\n{0};'.format(v_t))
        v_t.backward(Tensor(np.array([[1, 1], [1, 1], [1, 1]])))
        print('grad v: \r\n{0};'.format(v.grad))

