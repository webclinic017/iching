# 
import unittest
from biz.drlt.app_config import AppConfig
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.asset_actions import AssetActions
from biz.drlt.envs.minute_bar_env import State

class TMinuteBarEnv(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_State_main(self):
        print('生成环境状态类')
        year = 2016
        instrument = 'data\\YNDX_160101_161231.csv'
        stock_data = BarData.load_year_data(year)
        print('stock_data: {0};'.format(stock_data[instrument]))
        st = State(bars_count=10, commission_perc=0.1, reset_on_close=True, reward_on_close=True,volumes=True)
        st.reset(stock_data[instrument], offset=AppConfig.BARS_COUNT+1)
        obs = st.encode()
        print('initial observation: type:{0}; shape:{1};'.format(type(obs), obs))
        action = AssetActions.Buy
        reward, done = st.step(action=action)
        print('reward={0}; done={1};'.format(reward, done))
        obs = st.encode()
        info = {
            'instrument': 'YNDX',
            'offset': st._offset
        }
        print('********************** step **********************************')
        print('obs: {0};'.format(obs))
        print('info: {0};'.format(info))
        #while True:
        #    st.step()
        print('^_^')
        self.assertTrue(1>0)