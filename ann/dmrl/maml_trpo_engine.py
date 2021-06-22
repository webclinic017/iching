# 
import yaml

class MamlTrpoEngine(object):
    def __init__(self):
        self.name = 'ann.dmrl.maml_trpo_engine.MamlTrpoEngine'

    def startup(self, args={}):
        print('基于TRPO的元强化学习算法 v0.0.1')
        self.train(args=args)

    def train(self, args={}):
        with open(args['config'], 'r', encoding='utf-8') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        print('config: {0}; {1};'.format(type(config), config))