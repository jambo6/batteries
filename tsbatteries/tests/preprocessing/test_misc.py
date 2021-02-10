import torch

from tsbatteries.preprocessing import misc


def test_ragged_tensor_list_to_tensor():
    # This just calls pack padded sequence so we shouldnt have problems with it
    # Setup an unequal length tensor list
    list_arrays = [
        [[1, 2, 3], [3, 4, float("nan")]],
        [[1, 2, float("nan")], [5, 6, 1], [9, 10, 2]],
    ]
    tensor_list = [torch.tensor(x, dtype=float) for x in list_arrays]

    # True solution given fill
    def get_truth(fill_value):
        a1 = torch.cat([tensor_list[0], fill_value * torch.ones(1, 3)], dim=0)
        return torch.stack([a1, tensor_list[1]])

    for fill_value in [float("nan"), 0.0]:
        tensor, length = misc._pad_ragged_tensors(tensor_list, fill_value=fill_value)
        assert tensor.allclose(get_truth(fill_value), equal_nan=True)
