#

class IchingWriter(object):
    def __init__(self):
        self.name = 'ann.iching_writer.IchingWriter'

    def add_scalar(self, tag: str, scalar_value: 'Any', global_step: int, walltime: float=0.0) -> None:
        print('add_scalar: {0}={1} {2};'.format(tag, scalar_value, global_step))

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int, walltime: float=0.0) -> None:
        print('add_scalar: {0}={1} {2};'.format(main_tag, tag_scalar_dict, global_step))

    def close(self):
        print('close...')
