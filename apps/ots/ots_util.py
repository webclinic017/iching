#

class OtsUtil(object):
    STEP_THRESHOLD = 408
    step = 0

    @staticmethod
    def log(msg):
        if OtsUtil.step >= OtsUtil.STEP_THRESHOLD:
            print(msg)