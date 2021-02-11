import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (FunctionTransformer, MaxAbsScaler,
                                   MinMaxScaler, StandardScaler)

from ._mixin import apply_fit_to_channels, apply_transform_to_channels

SCALERS = {
    "stdsc": StandardScaler,
    "ma": MaxAbsScaler,
    "mms": MinMaxScaler,
}


class TensorScaler(TransformerMixin):
    """Scaling for 3D tensors.

    Assumes the size is (..., length, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.

    Arguments:
        method (str): Scaling method, one of ('stdsc', 'ma', 'mms').
        scaling_function (transformer): Specification of an sklearn transformer that performs a scaling operation.
            Only one of this or scaling can be specified.
    """

    def __init__(self, method="stdsc", scaling_function=None):
        self.scaling = method

        if all([method is None, scaling_function is None]):
            self.scaler = FunctionTransformer(func=None)
        elif isinstance(method, str):
            self.scaler = SCALERS.get(method)()
            assert (
                self.scaler is not None
            ), "Scalings allowed are {}, recieved {}.".format(SCALERS.keys(), method)
        else:
            self.scaler = scaling_function

    @apply_fit_to_channels
    def fit(self, data, labels=None):
        self.scaler.fit(data)
        return self

    @apply_transform_to_channels
    def transform(self, data):
        output_data = torch.Tensor(self.scaler.transform(data))
        return output_data


def scale_tensors(tensors, method="stdsc"):
    """A function version of TensorScaler, if multiple tensors are specified the first is used to fit the scaler

    Arguments:
        tensors (tensor or list of tensors): Data to transform. If multiple tensors are specified, the first is used to
            fit.
        method (str): See TensorScaler.

    Returns:
        Same format as input only now scaled.
    """
    scaler = TensorScaler(method=method)

    if isinstance(tensors, list):
        scaler.fit(tensors[0])
        output = [scaler.transform(x) for x in tensors]
    else:
        output = scaler.fit_transform(tensors)

    return output
