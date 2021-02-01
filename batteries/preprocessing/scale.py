import torch
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    FunctionTransformer,
)
from sklearn.base import TransformerMixin

SCALERS = {"stdsc": StandardScaler(), "ma": MaxAbsScaler(), "mms": MinMaxScaler()}


class TensorScaler(TransformerMixin):
    """Scaling for 3D tensors.

    Assumes the size is (..., length, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.
    """

    def __init__(self, method="stdsc", scaling_function=None):
        """
        Args:
            method (str): Scaling method, one of ('stdsc', 'ma', 'mms').
            scaling_function (transformer): Specification of an sklearn transformer that performs a scaling operation.
                Only one of this or scaling can be specified.
        """
        self.scaling = method

        if all([method is None, scaling_function is None]):
            self.scaler = FunctionTransformer(func=None)
        elif isinstance(method, str):
            self.scaler = SCALERS.get(method)
            assert (
                self.scaler is not None
            ), "Scalings allowed are {}, recieved {}.".format(SCALERS.keys(), method)
        else:
            self.scaler = scaling_function

    def _trick(self, data):
        return data.reshape(-1, data.shape[2])

    def _untrick(self, data, shape):
        return data.reshape(shape)

    def fit(self, data, y=None):
        self.scaler.fit(self._trick(data), y)
        return self

    def transform(self, data):
        scaled_data = self.scaler.transform(self._trick(data))
        output_data = torch.Tensor(self._untrick(scaled_data, data.shape))
        return output_data


def scale_tensors(tensors, method="stdsc"):
    """A function version of TensorScaler, if multiple tensors are specified the first is used to fit the scaler

    Args:
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
