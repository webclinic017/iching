#

class DmrlApp(object):
    def __init__(self):
        self.name = 'ann.dmrl.dmrl_app.DmrlApp'

    # python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder maml-halfcheetah-vel --seed 1 --num-workers 8
    def startup(self):
        print('元强化学习（meta-rl） v0.0.3')
        args = self.get_cmd_args()
        print('args: {0};'.format(args))
        print('^_^ The End ^_^')

    def get_args(self):
        import argparse
        import multiprocessing as mp
        parser = argparse.ArgumentParser(description='Reinforcement learning with '
                'Model-Agnostic Meta-Learning (MAML) - Train')
        parser.add_argument('--mode', type=str, required=True,
                help='the program to start')
        '''
        parser.add_argument('--config', type=str, required=True,
                help='path to the configuration file.')
        # Miscellaneous
        misc = parser.add_argument_group('Miscellaneous')
        misc.add_argument('--output-folder', type=str,
            help='name of the output folder')
        misc.add_argument('--seed', type=int, default=None,
            help='random seed')
        misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
            help='number of workers for trajectories sampling (default: '
                '{0})'.format(mp.cpu_count() - 1))
        misc.add_argument('--use-cuda', action='store_true',
            help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
            'is not guaranteed. Using CPU is encouraged.')
        return parser.parse_args()
        '''
        args = {}
        args['mode'] = 'dmrl'
        args['config'] = 'configs/maml/2d-navigation.yaml'
        args['misc'] = {}
        args['misc']['output-folder'] = './work'
        args['misc']['seed'] = 1
        args['misc']['num-workers'] = 1
        args['misc']['use-cuda'] = True
        return args
