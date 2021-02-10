import torch

from tsbatteries.preprocessing import impute


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
            x_ffilled_actual = impute._forward_fill(x)
            assert x_ffilled.allclose(x_ffilled_actual, equal_nan=True)
