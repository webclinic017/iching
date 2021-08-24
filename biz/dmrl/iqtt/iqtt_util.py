#
import os
import torch

class IqttUtil(object):
    @staticmethod
    def mask_(matrices, maskval=0.0, mask_diagonal=True):
        """
        Masks out all values in the given batch of matrices where i <= j holds,
        i < j if mask_diagonal is false
        In place operation
        :param tns:
        :return:
        """

        b, h, w = matrices.size()

        indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
        matrices[:, indices[0], indices[1]] = maskval

    @staticmethod
    def d(tensor=None):
        """
        Returns a device string either for the best available device,
        or for the device corresponding to the argument
        :param tensor:
        :return:
        """
        if tensor is None:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cuda' if tensor.is_cuda else 'cpu'

    @staticmethod
    def here(subpath=None):
        """
        :return: the path in which the package resides (the directory containing the 'former' dir)
        """
        if subpath is None:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

    @staticmethod
    def contains_nan(tensor):
        return bool((tensor != tensor).sum() > 0)