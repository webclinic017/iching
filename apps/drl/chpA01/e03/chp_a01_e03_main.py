# Sigmoid Regression example
from apps.drl.chpA01.e03.sigmoid_regression_engine import SigmoidRegressionEngine

class ChpA01E03Main(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('Sigmoid Regression Example')
        engine = SigmoidRegressionEngine()
        engine.exp()