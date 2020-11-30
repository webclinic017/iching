#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
import math

class VolatilityArbitrage(object):
    def __init__(self):
        self.refl = ''
        
    def startup(self):
        print('VolatilityArbitrage v0.0.3')
        self.ds_file = './data/50ETF.xlsx'
        etf_close = pd.read_excel(self.ds_file,"close")
        etf_ivx = pd.read_excel(self.ds_file,"ivx")
        etf_hv = pd.read_excel(self.ds_file,"hv30")
        lastdate = pd.read_excel(self.ds_file,"ETF_option_lasttradingdate")
        etf_option_name = pd.read_excel(self.ds_file,"at_money_name")
        
        ##填补缺失值
        for j in range(1,len(etf_close.columns.tolist())):
            for i in range(len(etf_close.date.values.tolist())-1,0,-1):
                if math.isnan(etf_close.iat[i,j]) and math.isnan(etf_close.iat[i-1,j]):
                    etf_close.iat[i,j]= etf_close.iat[i+1,j]
                elif math.isnan(etf_close.iat[i,j]) and math.isnan(etf_close.iat[i-1,j])==False:
                    etf_close.iat[i,j]=(etf_close.iat[i-1,j]+etf_close.iat[i+1,j])/2
                    
        fee = 5.0 # 手续费
        slippage = 5.0 # 滑点
        capital = 1000000.0 # 初始资金
        size=50 # straddle
        option_value=0 # 期权价值，初始时为0
        remain_money=capital # 当前资金
        total_money = [remain_money]
        trade_option = pd.DataFrame()
        ### 回测参数设置
        open_b = 1.5 # 
        close_b = 0.0001
        #记录交易日期，交易内容，交易posit
        d = []
        trade_content = []
        trade_posit = []
        #计算ivx
        #构造ivx曲线
        ivxl = []
        #计算ivx的函数
        #记录交易日期，交易内容，交易posit

        #构造长（20日）短（5日）均线
        #短均线
        ma5 = [0,0,0,0,0]
        a = 0
        for i in np.arange(5,905,1):
            ma5.append((etf_hv.hv[i-1]+etf_hv.hv[i-2]+etf_hv.hv[i-3]+etf_hv.hv[i-4]+etf_hv.hv[i-5])/5)
        #长均线    
        ma20 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        b = 0
        for i in np.arange(20,905,1):
            for j in np.arange(1,21):
                b += etf_hv.hv[i-j]
            ma20.append(b/20)
            b = 0


        #plot ma5 and ma20
        plt.title('Moving Average Lines')
        plt.plot(etf_hv.date, ma5, linestyle = '-.',color='black', label='MA5')
        plt.plot(etf_hv.date, ma20,color='red', label='MA20')
        plt.legend() 
        plt.xlabel('Date')
        plt.ylabel('Average hv')
        plt.savefig("MAL.png")
        plt.show()

        #计算ivx
        #构造ivx曲线


        ### 画出平值附近期权的隐波
        for date in etf_option_name.date.values:
            ivxl.append(self.cal_ivx(date, etf_option_name, etf_ivx))
            
                
        #plot ivx
        plt.title('ivx line')
        plt.plot(etf_option_name.date, ivxl, label = 'ivx')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('ivx')
        plt.savefig("ivx.png")
        plt.show()

        #连接成DataFrame
        c = {"number":range(905), "date":etf_hv.date, "ma5":ma5, "ma20":ma20}
        etf_hv1 = pd.DataFrame(c)

        ############
        ##计算hv的标准差(时段为20天)
        hv30plus=etf_hv['hv'].values.tolist()
        hv30_std = [(np.array(hv30plus)[i:i+20]).std() for i in range(len(hv30plus))]





        ########
        day=[]
        for date in etf_option_name.date.values:
            trade_option, remain_money = self.handle_ivx(date, lastdate, trade_option, size, remain_money, etf_close, d, trade_posit, trade_content, etf_hv, etf_option_name, etf_ivx, etf_hv1, open_b, close_b, total_money, fee, slippage)
            date=pd.to_datetime(str(date)).strftime('%Y-%m-%d') #date为字符串
            date = datetime.strptime(date, "%Y-%m-%d")
            day.append(date)

        DAY_MAX=len(total_money)


        backtest_return = total_money[-1] / capital - 1
        annulized, max_drawdown, rtn_vol, sharpe, sortino = self.cacul_performance(total_money, capital, DAY_MAX)


        ## 回测绩效与绘图
        print('Return: %.2f%%' % (backtest_return * 100.0))
        print('Annualized Return: %.2f%%' % (annulized * 100.0))
        print('Maximal Drawdown: %.2f%%' % (max_drawdown * 100.0))
        print('Annualized Vol: %.2f%%' % (100.0 * rtn_vol))
        print('Sharpe Ratio: %.4f' % sharpe)
        print('Sortino Ratio: %.4f' % sortino)


        #sns.set_style('white')
        plt.figure(figsize=(8, 5))
        plt.plot(day, total_money[1:])
        plt.xlabel('Date')
        plt.ylabel('Money')
        plt.title('Money Curve')
        plt.grid(True)
        plt.savefig("result.png")
        plt.show()
        
        
    def add_open(self, trade_option, num,call_name,put_name):
        #"正在开仓中的合约"
        if trade_option.empty:
            t = pd.Series([call_name,put_name,num], index=["call","put","size"])
            trade_option = trade_option.append(t, ignore_index=True)
        else:
            if call_name not in trade_option['call'].values:
                t = pd.Series([call_name,put_name,num], index=["call","put","size"])
                trade_option = trade_option.append(t, ignore_index=True)
        return trade_option
    
    def straddle(self, date, trade_option, size, posit,call_name,put_name, etf_close, d, trade_posit, trade_content, fee, slippage):#资金处理
        ##
        call_close = etf_close[etf_close.date==date][call_name].values[0]
        put_close = etf_close[etf_close.date==date][put_name].values[0]

        if posit=="buy": #买跨式期权
            num=size
            trade_option = self.add_open(trade_option, num,call_name,put_name)
            d.append(date)
            trade_posit.append("buy")
            trade_content.append('buy: ' + str(call_name) + ' and buy' + str(put_name))
            print(str(date) + 'buy: ' + str(call_name) + ' and buy ' + str(put_name))
            money_chg=-10000.0*size*call_close-10000.0*size*put_close
        elif posit=="sell":#卖跨式期权
            num=-size
            trade_option = self.add_open(trade_option, num,call_name,put_name)
            d.append(date)
            trade_posit.append("sell")
            trade_content.append( 'sell: ' + str(call_name) + ' and sell' + str(put_name))
            print(str(date) + 'sell: ' + str(call_name) + ' and sell ' + str(put_name))
            money_chg=10000.0*size*call_close+10000.0*size*put_close
        elif posit=="close buy":#冲销多头——卖出期权
            trade_option=trade_option[trade_option['call']!=call_name]
            d.append(date)
            trade_posit.append("close buy")
            trade_content.append( 'close buy: ' + str(call_name) + ' and sell' + str(put_name))
            print(str(date) + 'close buy: ' + str(call_name) + ' and close buy ' + str(put_name))
            money_chg=10000.0*size*call_close+10000.0*size*put_close
        elif posit=="close sell":#冲销空头——买入期权
            trade_option=trade_option[trade_option['call']!=call_name]
            d.append(date)
            trade_posit.append("close sell")
            trade_content.append( 'close sell: ' + str(call_name) + ' and buy' + str(put_name))
            print(str(date) + 'close sell: ' + str(call_name) + ' and close sell ' + str(put_name))
            money_chg=-10000.0*size*call_close-10000.0*size*put_close

        return money_chg - 2 * size * fee - 2 * size * slippage / 2.0, trade_option, size
        
    #计算ivx的函数
    def cal_ivx(self, date, etf_option_name, etf_ivx):
        call_name=etf_option_name[etf_option_name.date==date]['call'].values[0]
        put_name=etf_option_name[etf_option_name.date==date]['put'].values[0]
        call_ivx = etf_ivx[etf_ivx.date==date][call_name].values[0]
        put_ivx = etf_ivx[etf_ivx.date==date][put_name].values[0]
        return ((call_ivx + put_ivx )/2)
        
    ##开仓平仓交易
    def handle_ivx(self, date, lastdate, trade_option, size, remain_money, etf_close, d, trade_posit, trade_content,
                etf_hv, etf_option_name, etf_ivx, etf_hv1, open_b, close_b, total_money, fee, slippage):
        hv = etf_hv[etf_hv.date==date]['hv'].values[0]
        hvstd = etf_hv[etf_hv.date==date]['hv_std'].values[0]
        call_name=etf_option_name[etf_option_name.date==date]['call'].values[0]
        put_name=etf_option_name[etf_option_name.date==date]['put'].values[0]
        call_ivx = etf_ivx[etf_ivx.date==date][call_name].values[0]
        put_ivx = etf_ivx[etf_ivx.date==date][put_name].values[0]
        ivx= (call_ivx + put_ivx )/2
        option_value=0
        number = int(etf_hv1[etf_hv1.date==date].number)
        
        ### 开仓
        if trade_option.empty:
            if ivx > hv + open_b * hvstd and (ivx>etf_hv1.ma20[number]) :
                call_close = etf_close[etf_close.date == date][call_name].values[0]
                put_close = etf_close[etf_close.date == date][put_name].values[0]
                posit = 'sell'
                change, trade_option, size = self.straddle(date, trade_option, size, posit,call_name,put_name, etf_close, d, trade_posit, trade_content, fee, slippage)
                option_value = option_value - 10000.0 * size * call_close - 10000.0 * size * put_close
            elif ivx < hv - open_b * hvstd and (ivx<etf_hv1.ma20[number]):
                call_close = etf_close[etf_close.date == date][call_name].values[0]
                put_close = etf_close[etf_close.date == date][put_name].values[0]
                posit = 'buy'
                change, trade_option, size = self.straddle(date, trade_option, size, posit,call_name,put_name, etf_close, d, trade_posit, trade_content, fee, slippage)
                option_value = option_value + 10000.0 * size * call_close + 10000.0 *size * put_close
            else:
                change=0
        else:
            ###平仓
            for call_name in trade_option['call']:
                num=trade_option[trade_option['call']==call_name]["size"].values.tolist()[0]
                put_name=trade_option[trade_option['call']==call_name]["put"].values.tolist()[0]

                if ( hv-close_b *hvstd < ivx < hv + close_b *hvstd or self.expire(call_name,date, lastdate)=="T" )and num >0:
                    posit = 'close buy'
                    change, trade_option, size = self.straddle(date, trade_option, size, posit,call_name,put_name, etf_close, d, trade_posit, trade_content, fee, slippage)
                elif (hv < ivx < etf_hv1.ma20[number]  or self.expire(call_name,date, lastdate)=="T" )and num <0:
                    posit = 'close sell'
                    change, trade_option, size = self.straddle(date, trade_option, size, posit,call_name,put_name, etf_close, d, trade_posit, trade_content, fee, slippage)
                else:
                    call_close = etf_close[etf_close.date == date][call_name].values[0]
                    put_close = etf_close[etf_close.date == date][put_name].values[0]
                    option_value=option_value+10000.0*num*call_close+10000.0*num*put_close
                    change=0
        remain_money += change
        total_money.append(remain_money + option_value)
        return trade_option, remain_money
        
    ########判断看涨期权是否到期的函数
    def expire(self, call_name,date, lastdate):
        if date in lastdate.lasttradingdate.values and (call_name in lastdate[lastdate.lasttradingdate == date]['symbol'].values):
            expireTF="T"
        else:
            expireTF="F"
        return expireTF
        
    def cacul_performance(self, total_money, capital, DAY_MAX):
        """
        Calculate annualized return, Sharpe ratio, maximal drawdown, return volatility and sortino ratio
        :return: annualized return, Sharpe ratio, maximal drawdown, return volatility and sortino ratio
        """
        rtn = total_money[-1] / capital - 1

        annual_rtn = np.power(rtn + 1, 252.0 / DAY_MAX) - 1  # 复利
        annual_rtn = rtn * 252 / DAY_MAX  # 单利
        print(total_money)
        annual_lst = [(total_money[k + 1] - total_money[k]) / total_money[k] for k in range(DAY_MAX - 1)]
        annual_vol = np.array(annual_lst).std() * np.sqrt(252.0)

        rf = 0.04

        semi_down_list = list(filter(lambda x: True if x < rf/252  else False, annual_lst))
        # semi_down_list = [annual_lst[k] < rf / 252 for k in range(trade_period - 1)]
        semi_down_vol = np.array(semi_down_list).std() * np.sqrt(252)
        sharpe_ratio = (annual_rtn - rf) / annual_vol
        sortino_ratio = (annual_rtn - rf) / semi_down_vol

        max_drawdown_ratio = 0
        for e, i in enumerate(total_money):
            for f, j in enumerate(total_money):
                if f > e and float(j - i) / i < max_drawdown_ratio:
                    max_drawdown_ratio = float(j - i) / i

        return annual_rtn, max_drawdown_ratio, annual_vol, sharpe_ratio, sortino_ratio