#
from apps.nup_app import NupApp
import sys
import getopt
import numpy as np
'''
from apps.ogml.ogml_app import OgmlApp
#from apps.tp.tp_app import TpApp
from apps.rxgb.rxgb_app import RxgbApp
from apps.asml.asml_app import AsmlApp
from apps.asdk.asdk_app import AsdkApp
from apps.tcv.tcv_app import TcvApp
from apps.fmml.fmml_app import FmmlApp
from apps.wxs.wxs_app import WxsApp
'''
from apps.nup_app import NupApp
from apps.sop.sop_app import SopApp
from apps.ots.ots_app import OtsApp
from apps.drl.drl_app import DrlApp
#from iqt.iqt_app import IqtApp


def norm_batch_tasks(batch_vals, task_num):
    arrs = []
    batch_size = len(batch_vals) // task_num
    for i in range(task_num):
        arrs.append(np.array(batch_vals[i*batch_size : (i+1)*batch_size]).reshape((batch_size,1)))
    return np.hstack(tuple(arrs)).mean(axis=1)

def exp():
    pass

params = {
    'version': '0.0.1',
    'mode': '1',
    'exp': ''
}

def main(argv={}):
    opts, args = getopt.getopt(argv, 'v:i', ['mode=', 'exp='])
    print('args:{0}; {1};'.format(type(args), args))
    for opt, val in opts:
        if opt=='-i':
            print('易经量化交易平台')
        elif opt=='-v':
            params['version'] = val
        elif opt=='--mode':
            params['mode'] = val
        elif opt=='--exp':
            params['exp'] = val
    print('mode={0}; exp={1};'.format(params['mode'], params['exp']))
    #app = OgmlApp()
    #app = TpApp()
    #app = RxgbApp()
    #app = AsmlApp()
    #app = AsdkApp()
    #app = TcvApp()
    #app = FmmlApp()
    #app = SopApp()
    if params['mode'] == 'ots':
        app = OtsApp()
    elif params['mode'] == 'drl':
        app = DrlApp()
    #elif params['mode'] == 'iqt':
    #    app = IqtApp()
    else:
        app = NupApp()
    app.startup()

if '__main__' == __name__:
    '''
    命令行启动参数：python app_main.py -i -v 0.0.1 --mode iqt --exp tt a1 a2 a3
      短选项：-字母 -字母:（接收参数） -t
      长选项：--字符串
      其他参数
    '''
    argv = sys.argv[1:]
    main(argv)