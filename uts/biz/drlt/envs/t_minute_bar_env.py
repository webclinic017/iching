# 
import unittest
from biz.drlt.app_config import AppConfig
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.asset_actions import AssetActions
from biz.drlt.envs.minute_bar_env import State
from biz.drlt.envs.minute_bar_env import MinuteBarEnv

class TMinuteBarEnv(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_MinuteBarEnv_main(self):
        '''
        研究市场环境类
        '''
        year = 2016
        instrument = 'data\\YNDX_160101_161231.csv'
        stock_data = BarData.load_year_data(year)
        print('stock_data: {0};'.format(stock_data[instrument]))
        env = MinuteBarEnv(
                stock_data, bars_count=AppConfig.BARS_COUNT, volumes=True)
        obs = env.reset()
        seq = 1
        while True:
            if seq > 3:
                done = True
            if 1 == seq:
                action = AssetActions.Buy
            elif 2 == seq:
                action = AssetActions.Sell
            else:
                action = AssetActions.Skip
            obs, reward, done, info = env.step(action)
            if done:
                break
            env.render(mode='human', obs=obs, reward=reward, info=info)
            seq += 1
        print('observation: {0};'.format(obs))

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
        # 购买股票
        action = AssetActions.Buy
        reward, done = st.step(action=action)
        obs = st.encode()
        info = {
            'instrument': 'YNDX',
            'offset': st._offset
        }
        self._print_State_step_result(reward, done, obs, info)
        # 持有
        action = AssetActions.Skip
        reward, done = st.step(action=action)
        obs = st.encode()
        info = {
            'instrument': 'YNDX',
            'offset': st._offset
        }
        self._print_State_step_result(reward, done, obs, info)
        # 卖出
        action = AssetActions.Sell
        reward, done = st.step(action=action)
        obs = st.encode()
        info = {
            'instrument': 'YNDX',
            'offset': st._offset
        }
        self._print_State_step_result(reward, done, obs, info)
        print('^_^')
        self.assertTrue(1>0)

    def _print_State_step_result(self, reward, done, obs, info):
        '''
        打印经过一次step后情况，包括：奖励、是否完成、当前状态、额外信息
        '''
        print('reward={0}; done={1}; info={2};'.format(reward, done, info['offset']))
        print('当前观测状态: {0};'.format(obs))
        print('**********************************************************************************************')