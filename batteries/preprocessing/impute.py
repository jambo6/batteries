import torch
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from .mixin import apply_fit_to_channels, apply_transform_to_channels
from batteries.misc import forward_fill


class BasicImpute(TransformerMixin):
    """Basic imputation for tensors. Simply borrows from sklearns SimpleImputer.

    Assumes the size is (..., length, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.
    """

    def __init__(self, strategy, fill_value):
        """
        See sklearn.impute.SimpleImputer.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    @apply_fit_to_channels
    def fit(self, data, labels=None):
        self.imputer.fit(data)
        return self

    @apply_transform_to_channels
    def transform(self, data):
        output_data = torch.Tensor(self.imputer.transform(data))
        return output_data


class NegativeImputer(TransformerMixin):
    """ Replace negative values with zero. """
    def __init__(self, fill_value=0.):
        self.fill_value = fill_value

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        data[data < 0] = self.fill_value
        return data


class ForwardFill(TransformerMixin):
    """ Forward fill the data along the length index. """
    def __init__(self, length_index=2, backfill=False):
        """
        Args:
            length_index (int): Set the index of the data for which to perform the fill. The default is -2 due to the
                standard (..., length, input_channels) format.
            backfill (bool): Set True to perform a backwards fill.
        """
        self.length_index = length_index
        if backfill:
            raise NotImplementedError

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        return forward_fill(data, fill_index=self.length_index)
