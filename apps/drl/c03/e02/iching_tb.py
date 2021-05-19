#
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class IchingTb(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.grid()
        self.xdata, self.ydata = [], []

    def startup(self, args={}):
        self.ani = animation.FuncAnimation(self.fig, self.draw_frame, self.data_generator, interval=10, init_func=self.init_plt)
        plt.show()

    def init_plt(self) -> 'Any':
        '''
        Matplotlib动画初始化方法
        '''
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 10)
        del self.xdata[:]
        del self.ydata[:]
        self.line.set_data(self.xdata, self.ydata)
        return self.line,

    def data_generator(self) -> 'obj':
        for cnt in itertools.count():
            t = cnt / 10
            yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)

    def draw_frame(self, frame) -> 'Any':
        # update the data
        t, y = frame
        self.xdata.append(t)
        self.ydata.append(y)
        xmin, xmax = self.ax.get_xlim()
        if t >= xmax:
            self.ax.set_xlim(xmin, 2*xmax)
            self.ax.figure.canvas.draw()
        self.line.set_data(self.xdata, self.ydata)
        return self.line,