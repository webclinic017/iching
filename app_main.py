#
import sys
import getopt
import numpy as np
from argparse import ArgumentParser
from apps.nup_app import NupApp
'''
#from apps.tp.tp_app import TpApp
from apps.rxgb.rxgb_app import RxgbApp
from apps.asml.asml_app import AsmlApp
from apps.asdk.asdk_app import AsdkApp
from apps.ogml.ogml_app import OgmlApp # MAML元学习
from apps.tcv.tcv_app import TcvApp
from apps.fmml.fmml_app import FmmlApp
from apps.wxs.wxs_app import WxsApp
'''
from apps.nup_app import NupApp
#from apps.sop.sop_app import SopApp
from apps.ots.ots_app import OtsApp
from apps.drl.drl_app import DrlApp
from apps.fxcm.fxcm_app import FxcmApp
#from biz.dct.dct_app import DctApp
from biz.drlt.drlt_app import DrltApp
from apps.mml.mml_app import MmlApp
from apps.dmrl.maml.maml_app import MamlApp
from ann.dmrl.dmrl_app import DmrlApp
from biz.dmrl.dmrl_main import DmrlMain
from apps.fmts.fmts_app import FmtsApp

# 启动命令行参数默认值
params = {
    'version': '0.0.1',
    'mode': '1',
    'exp': ''
}
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', dest='run_mode', default='nup', type=str)
    parser.add_argument("-e", "--num-epochs",
                    dest="num_epochs",
                    help="Number of epochs.",
                    default=80, type=int)
    return vars(parser.parse_args())

def main(argv={}):
    # 解析命令行参数
    args = parse_args()
    # 根据mode选项相应的应用
    if args['run_mode'] == 'ots': # 期权交易系统
        app = OtsApp()
    elif args['run_mode'] == 'drl': # 深度强化学习理论学习
        app = DrlApp()
    #elif params['mode'] == 'dct':
    #    app = DctApp()
    elif args['run_mode'] == 'fxcm': # Python for finance
        app = FxcmApp()
    elif args['run_mode'] == 'drlt': # DQN用于量化交易
        app = DrltApp()
    elif args['run_mode'] == 'mml': # 机器学习中的数学
        app = MmlApp()
    elif args['run_mode'] == 'maml':
        app = MamlApp()
    elif args['run_mode'] == 'dmrl':
        app = DmrlApp()
    elif args['run_mode'] == 'bizdmrl':
        app = DmrlMain()
    elif args['run_mode'] == 'fmts':
        app = FmtsApp()
    else:
        app = NupApp()
    app.startup(args)

if '__main__' == __name__:
    '''
    命令行启动参数示例：python app_main.py -i -v 0.0.1 --mode iqt --exp tt a1 a2 a3
    '''
    argv = sys.argv[1:]
    main(argv)




def norm_batch_tasks(batch_vals, task_num):
    arrs = []
    batch_size = len(batch_vals) // task_num
    for i in range(task_num):
        arrs.append(np.array(batch_vals[i*batch_size : (i+1)*batch_size]).reshape((batch_size,1)))
    return np.hstack(tuple(arrs)).mean(axis=1)

def exp():
    #app = OgmlApp()
    #app = TpApp()
    #app = RxgbApp()
    #app = AsmlApp()
    #app = AsdkApp()
    #app = TcvApp()
    #app = FmmlApp()
    #app = SopApp()
    pass