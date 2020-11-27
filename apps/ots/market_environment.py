#

class MarketEnvironment(object):
    ''' 用于估值的市场环境相关信息，包括相关参数 '''
    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.consts = {} # 常数
        self.lists = {} # 列表
        self.curves = {} # 曲线，如贴现率

    def add_const(self, key, val):
        self.consts[key] = val

    def get_const(self, key):
        if key in self.consts:
            return self.consts[key]
        else:
            return None

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        if key in self.lists:
            return self.lists[key]
        else:
            return None

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        if key in self.curves:
            return self.curves[key]
        else:
            return None

    def add_environment(self, env):
        ''' 若某个值已经存在则进行覆盖 '''
        for key in env.consts:
            self.consts[key] = env.consts[key]
        for key in env.lists:
            self.lists[key] = env.lists[key]
        for key in env.curves:
            self.curves[key] = env.curves[key]