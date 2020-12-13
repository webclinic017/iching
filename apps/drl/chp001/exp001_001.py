#

class Exp02001(object):
    def __init__(self):
        self.refl = ''

    def startup(self):
        self.bandit_walk()

    def bandit_walk(self):
        P = {
            0: {
                0: [(1.0, 0, 0.0, True)],
                1: [(1.0, 0, 0.0, True)]
            },
            1: {
                0: [(1.0, 0, 0.0, True)],
                1: [(1.0, 2, 1.0, True)]
            },
            2: {
                0: [(1.0, 2, 0.0, True)],
                1: [(1.0, 2, 0.0, True)]
            }
        }
        print(P)