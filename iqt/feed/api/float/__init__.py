
from typing import List

from iqt.feed.core import Stream
from iqt.feed.core.methods import Methods
from iqt.feed.core.mixins import DataTypeMixin


@Stream.register_accessor(name="float")
class FloatMethods(Methods):
    ...


@Stream.register_mixin(dtype="float")
class FloatMixin(DataTypeMixin):
    ...


class Float:
    """A class to register accessor and instance methods."""

    @classmethod
    def register(cls, names: List[str]):
        """A function decorator that adds accessor and instance methods for
        specified data type.

        Parameters
        ----------
        names : `List[str]`
            A list of names used to register the function as a method.

        Returns
        -------
        Callable
            A decorated function.
        """
        def wrapper(func):
            FloatMethods.register_method(func, names)
            FloatMixin.register_method(func, names)
            return func
        return wrapper


from iqt.feed.api.float.window import *
from iqt.feed.api.float.accumulators import *
from iqt.feed.api.float.imputation import *
from iqt.feed.api.float.operations import *
from iqt.feed.api.float.ordering import *
from iqt.feed.api.float.utils import *
