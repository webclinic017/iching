#
import matplotlib.pyplot as plt

class IchingTensorBoard(object):
    '''
    绘制深度学习模型训练中Loss变化曲线的工具类，更详细例子见apps/drl/c03/e03
    '''
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.grid()
        self.xdata, self.ydata = [], []
        self.current_step = 0
        plt.ion() # 启动交互模式

    def update_plot(self, x, y):
        self.xdata.append([x])
        self.ydata.append([y])
        plt.plot(self.xdata, self.ydata)
        plt.pause(0.01)

    def reset_plot(self):
        del self.xdata[:]
        del self.ydata[:]

    def stop_plot(self):
        plt.show(block=True)