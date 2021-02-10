import torch
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer as _SimpleImputer
from torchcde import linear_interpolation_coeffs

from tsbatteries.misc import forward_fill

from ._mixin import apply_fit_to_channels, apply_transform_to_channels


class BasicImpute(TransformerMixin):
    """Basic imputation for tensors. Simply borrows from sklearns SimpleImputer.

    Assumes the size is (..., length, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.

    Arguments:
        strategy (str): One of ('mean', 'median', 'most_frequent', 'constant').
        fill_value (float): The value to fill nans with, this is active only if `strategy = 'constant'`.
    """

    def __init__(self, strategy, fill_value):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = _SimpleImputer(strategy=strategy, fill_value=fill_value)

    @apply_fit_to_channels
    def fit(self, data, labels=None):
        self.imputer.fit(data)
        return self

    @apply_transform_to_channels
    def transform(self, data):
        output_data = torch.Tensor(self.imputer.transform(data))
        return output_data


class NegativeImputer(TransformerMixin):
    """Replace negative values with zero.

    Arguments:
        fill_value (float): The values to replace the negative values with.
    """

    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        data[data < 0] = self.fill_value
        return data


class ForwardFill(TransformerMixin):
    """Forward fill the data along the length index.

    Arguments:
        length_index (int): Set the index of the data for which to perform the fill. The default is -2 due to the
            standard (..., length, input_channels) format.
        backfill (bool): Set True to perform a backwards fill.
    """

    def __init__(self, length_index=2, backfill=False):
        self.length_index = length_index
        if backfill:
            raise NotImplementedError

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        return forward_fill(data, fill_index=self.length_index)


class LinearInterpolation(TransformerMixin):
    """Perform linear (or rectilinear) interpolation on the missing values.

    Arguments:
        rectilinear (bool): Set True for rectilinear interpolation. Note that this will result in a tensor of length
            (2 * length - 1), approximately double.
    """

    def __init__(self, rectilinear=False):
        self.rectilinear = rectilinear

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        rectilinear = 0 if self.rectilinear else None
        return linear_interpolation_coeffs(data, rectilinear=rectilinear)
