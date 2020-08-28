# 应用注册表测试类
import unittest
from apps.sop.sop_registry import SopRegistry

class TSopRegistry(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_put(self):
        key = 'age'
        val = 102
        SopRegistry.put(key, val)
        print('val: {0};'.format(SopRegistry.PARAMS[key]))

    def test_get(self):
        key = 'age'
        val = 102
        SopRegistry.put(key, val)
        print('get age: {0};'.format(SopRegistry.get(key)))