# 层定义

class Layer(object):
    def __init__(self):
        self.parameters = list()
        
    def get_parameters(self):
        return self.parameters