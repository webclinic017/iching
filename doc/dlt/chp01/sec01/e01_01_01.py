#
import numpy as np
import matplotlib.pyplot as plt

class E010101(object):
    def __init__(self):
        self.name = ''

    def startup(self):
        print('第1章 预备知识 第1节 高斯积分 例1 类高斯函数图像')
        x = np.arange(-10.0, 10.0, 0.01)
        y = np.exp(-x*x/2)
        fig, ax = plt.subplots()
        plt.plot(x, y)
        ax.set(xlabel='x', ylabel='y',
            title='raw gaussian')
        ax.grid()
        #fig.savefig('./work/p010101.png')
        plt.show()

def main(args={}):
    exp = E010101()
    exp.startup()

if '__main__' == __name__:
    main()