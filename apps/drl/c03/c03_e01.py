#
import math
#from tensorboardX import SummaryWriter

class C03E01(object):
    def __init__(self):
        self.name = 'apps.drl.c03.c03_e01.C03E01'

    def startup(self, args={}):
        print('检查tensorboardx')
        '''
        writer = SummaryWriter()
        funcs = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan
        }
        for angle in range(-360, 360):
            angle_rad = angle * math.pi / 180.0
            for name, func in funcs.items():
                val = func(angle_rad)
                writer.add_scalar(name, val, angle)
        writer.close()
        '''