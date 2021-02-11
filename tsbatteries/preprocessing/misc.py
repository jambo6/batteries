import torch
from sklearn.base import TransformerMixin
from torch.nn.utils.rnn import pad_sequence


class PadRaggedTensors(TransformerMixin):
    """Converts a list of unequal length tensors (or arrays) to a stacked tensor.

    This is done by extending tensors to the same length as the tensor in the list with maximal length.

    Arguments:
        fill_value (float): Value to fill if an array is extended.

    Returns:
        Two tensors, the first is of shape (len(tensor_list), max length from tensor_list, ...) and is the now stacked
            tensor, the second is the lengths of the tensors.
    """

    def __init__(self, fill_value=float("nan"), max_seq_len=None):
        self.fill_value = fill_value
        self.max_seq_len = max_seq_len

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        """Pads a list of tensors to max length.

        Args:
            data (list):

        Returns:

        """
        padded_tensor, _ = _pad_ragged_tensors(data, self.fill_value, self.max_seq_len)
        return padded_tensor


def _pad_ragged_tensors(tensor_list, fill_value=float("nan"), max_seq_len=None):
    if not isinstance(tensor_list[0], torch.Tensor):
        tensor_list = [torch.tensor(t) for t in tensor_list]

    # Reduce size
    if max_seq_len is not None:
        tensor_list = [x[:max_seq_len] for x in tensor_list]

    # Pad with a value that doesnt exist in the dataframe
    lengths = [len(x) for x in tensor_list]
    padded_tensor = pad_sequence(
        tensor_list, batch_first=True, padding_value=fill_value
    )

    return padded_tensor, lengths
