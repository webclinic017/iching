# Akshare数据下载和处理程序
import os
import akshare as ak
import numpy as np

class AkshareDataSource(object):
    DATA_FOLDER = 'apps/fmts/data/'
    def __init__(self):
        self.name = 'apps.fmts.ds.akshare_data_source.AkshareDataSource'

    @staticmethod
    def get_minute_bars(stock_symbol, period='1', adjust='hfq'):
        '''
        调用AkshareDataSource接口，默认为1分钟，复权后数据，保持历史价格不变，当配股、分拆、合并、派发股息后价格会变化，
        这种方式虽然不利于看盘，但是能反映真实收益率，量化交易研究中通常采用。
        '''
        data_file = '{0}{1}_1m.csv'.format(AkshareDataSource.DATA_FOLDER, stock_symbol)
        if not os.path.exists(data_file):
            # 如果文件不存在，则下载数据并保存到文件
            data = ak.stock_zh_a_minute(symbol=stock_symbol, period=period, adjust=adjust)
            data.to_csv(data_file)
        return AkshareDataSource.get_quotation_from_csv(data_file)

    @staticmethod
    def get_quotation_from_csv(csv_file): #csv_file = './data/aks_1ms/{0}_1m.csv'.format(stock_symbol)
        items = []
        with open(csv_file, 'r', encoding='utf-8') as fd:
            is_first_row = True
            for row in fd:
                if is_first_row:
                    is_first_row = False
                    continue
                row = row.strip()
                arrs0 = row.split(',')
                if len(row)<=0 or arrs0[1]=='' or arrs0[2]=='' or arrs0[3]=='' or arrs0[4]=='' or arrs0[5]=='':
                    break
                item = []
                item.append(str(arrs0[1]))
                item.append(float(arrs0[2]))
                item.append(float(arrs0[3]))
                item.append(float(arrs0[4]))
                item.append(float(arrs0[5]))
                item.append(float(arrs0[6]))
                items.append(item)
        return items