# 
import os
import json
import yaml

class MamlTrpoEngine(object):
    def __init__(self):
        self.name = 'ann.dmrl.maml_trpo_engine.MamlTrpoEngine'

    def startup(self, args={}):
        print('基于TRPO的元强化学习算法 v0.0.1')
        self.train(args=args)

    def train(self, args={}):
        # 读入配置文件
        with open(args['config'], 'r', encoding='utf-8') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        # 处理输出目录
        if config['output_folder'] is not None:
            if not os.path.exists(config['output_folder']):
                os.makedirs(config['output_folder'])
            policy_filename = os.path.join(config['output_folder'], 'policy.th')
            config_filename = os.path.join(config['output_folder'], 'config.json')
            with open(config_filename, 'w') as fd:
                #config.update(vars(args))
                json.dump(config, fd, indent=4)