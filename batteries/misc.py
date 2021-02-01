import torch


def forward_fill(x, fill_index=-2):
    """Forward fills data in a torch tensor of shape (..., length, input_channels) along the length dim.
    Arguments:
        x: tensor of values with first channel index being time, of shape (..., length, input_channels), where ... is
            some number of batch dimensions.
        fill_index: int that denotes the index to fill down. Default is -2 as we tend to use the convention (...,
            length, input_channels) filling down the length dimension.
    Returns:
        A tensor with forward filled data.
    """
    # Checks
    assert isinstance(x, torch.Tensor)
    assert x.dim() >= 2

    mask = torch.isnan(x)
    if mask.any():
        cumsum_mask = (~mask).cumsum(dim=fill_index)
        cumsum_mask[mask] = 0
        _, index = cumsum_mask.cummax(dim=fill_index)
        x = x.gather(dim=fill_index, index=index)

    return x
