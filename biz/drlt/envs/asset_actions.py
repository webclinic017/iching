# 单一金融资产的行动定义
import enum

class AssetActions(enum.Enum):
    Keep = 0
    Buy = 1
    Sell = 2

'''
另一种方案：
    Keep = 0
    Buy01 = 1 # 拿当前现金的10%来买
    Buy02 = 2 # 
    Buy03 = 3
    Buy04 = 4
    Buy05 = 5
    Buy06 = 6
    Buy07 = 7
    Buy08 = 8
    Buy09 = 9
    Buy10 = 10
    Sell01 = 11 # 卖出当前持有资产的10%
    Sell02 = 12
    Sell03 = 13
    Sell04 = 14
    Sell05 = 15
    Sell06 = 16
    Sell07 = 17
    Sell08 = 18
    Sell09 = 19
    Sell10 = 20 # 卖出当前持有资产的100%
'''