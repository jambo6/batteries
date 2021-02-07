import torch

from tsbatteries import misc


def test_forward_fill():
    """ Test forward fill against the results returned from a nested for loop. """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        # Check ffill
        for N, L, C in [(1, 5, 3), (2, 2, 2), (3, 2, 1)]:
            x = torch.randn(N, L, C).to(device)
            # Drop mask
            tensor_num = x.numel()
            mask = torch.randperm(tensor_num)[: int(0.3 * tensor_num)].to(device)
            x.view(-1)[mask] = float("nan")
            x_ffilled = x.clone().float()
            for i in range(0, x.size(0)):
                for j in range(x.size(1)):
                    for k in range(x.size(2)):
                        non_nan = x_ffilled[i, : j + 1, k][
                            ~torch.isnan(x[i, : j + 1, k])
                        ]
                        input_val = (
                            non_nan[-1].item() if len(non_nan) > 0 else float("nan")
                        )
                        x_ffilled[i, j, k] = input_val
            x_ffilled_actual = misc.forward_fill(x)
            assert x_ffilled.allclose(x_ffilled_actual, equal_nan=True)


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
        tensor, length = misc.ragged_tensor_list_to_tensor(
            tensor_list, fill_value=fill_value
        )
        assert tensor.allclose(get_truth(fill_value), equal_nan=True)
