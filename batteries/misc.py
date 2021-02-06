import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def forward_fill(x, fill_index=-2, backwards=False):
    """Forward fills data in a torch tensor of shape (..., length, input_channels) along the length dim.

    Arguments:
        x (tensor): Values with first channel index being time, of shape (..., length, input_channels), where ... is
            some number of batch dimensions.
        fill_index (int): Denotes the index to fill down. Default is -2 as we tend to use the convention (..., length,
            input_channels) filling down the length dimension.
        backwards (bool): Set True to first flip the tensor along the length axis so as to perform a backwards fill.

    Example:
        >>> x = torch.tensor([[1, 2], [float('nan'), 1], [2, float('nan')]], dtype=torch.float)
        >>> forward_fill(x, fill_index=-2, backwards=False)
        tensor([
            [1., 2.],
            [1., 1.],
            [2., 1.]
        ])

    Returns:
        A tensor with forward filled data.
    """
    # Checks
    assert isinstance(x, torch.Tensor)
    assert x.dim() >= 2

    # flipping if a backwards fill
    def backflip(x):
        x = x.flip(fill_index) if backwards else x
        return x

    x = backflip(x)

    mask = torch.isnan(x)
    if mask.any():
        cumsum_mask = (~mask).cumsum(dim=fill_index)
        cumsum_mask[mask] = 0
        _, index = cumsum_mask.cummax(dim=fill_index)
        x = x.gather(dim=fill_index, index=index)

    # Re-flip if backwards
    x = backflip(x)

    return x


def ragged_tensor_list_to_tensor(
    tensor_list, fill_value=float("nan"), max_seq_len=None
):
    """Converts a list of unequal length tensors (or arrays) to a stacked tensor.

    This is done by extending tensors to the same length as the tensor in the list with maximal length.

    Arguments:
        tensor_list (list or numpy object): List containing ragged tensors.
        fill_value (float): Value to fill if an array is extended.

    Returns:
        Two tensors, the first is of shape (len(tensor_list), max length from tensor_list, ...) and is the now stacked
            tensor, the second is the lengths of the tensors.
    """
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


def get_num_params(model):
    """ Gets the number of trainable parameters in a pytorch model. """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
