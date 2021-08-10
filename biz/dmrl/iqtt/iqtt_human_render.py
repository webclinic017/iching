#

class IqttHumanRender(object):
    def __init__(self):
        self.name = 'biz.dmrl.iqtt.iqtt_human_render.IqttHumanRender'

    def render(self, trades={}):
        print('### ^_^ ### {0}：bar({1}, {2}, {3}, {4}), state=(余额：{5}, 仓位：{6}, 净值：{7})'.format(
                trades['info']['current_step'], trades['obs'][0][50], trades['obs'][0][51], trades['obs'][0][52], trades['obs'][0][53],
                trades['info']['balance'], trades['info']['position'], trades['info']['net_value']
            ))