#

class ValuationBase(object):
    ''' 估值基类 '''
    def __init__(self, name, underlying, mkt_env, payoff_func):
        self.name = name
        self.pricing_date = mkt_env.pricing_date
        self.strike = mkt_env.get_const('strike') # 期权执行价格
        self.maturity = mkt_env.get_const('maturity') # 到期时间
        self.currency = mkt_env.get_const('currency')
        self.frequency = underlying.frequency # 标的
        self.paths = underlying.paths
        self. discount_curve = underlying.discount_curve 
        self.payoff_func = payoff_func
        self.underlying = underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

    def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)
        if volatility is not None:
            self.underlying.update(volatility=volatility)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            if not maturity in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
        self.underlying.instrument_values = None

    def calculate_delta(self, interval=None, accuracy=4):
        if interval is None:
            interval = self.underlying.initial_value / 50.0
        value_left = self.present_value(fixed_seed=True)
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        value_right = self.present_value(fixed_seed=True)
        self.underlying.update(initial_value=initial_del - interval)
        delta = (value_right - value_left) / interval
        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)

    def present_value(self, fixed_seed=True):
        return 0.0

    def calculate_vega(self, interval=0.01, accuracy=4):
        if interval < self.underlying.volatility/50.0:
            interval = self.underlying.volatility / 50.0
        value_left = self.present_value(fixed_seed=True)
        volatility_del = self.underlying.volatility + interval
        self.underlying.update(volatility=volatility_del)
        value_right = self.present_value(fixed_seed=True)
        self.underlying.update(volatility=volatility_del - interval)
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)