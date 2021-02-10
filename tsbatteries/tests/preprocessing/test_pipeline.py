import torch

from tsbatteries import preprocessing


def test_pipeline():
    # Create random tensor data
    data = [torch.randn(5, 2) for _ in range(5)]

    # Build a pipeline that should return no nans
    pipeline = preprocessing.Pipeline(
        [
            ("pad", preprocessing.PadRaggedTensors()),
            ("negative_impute", preprocessing.NegativeFilter()),
            ("stdsc", preprocessing.TensorScaler(method="stdsc")),
            ("interpolation", preprocessing.Interpolation(method="linear")),
            ("backfill", preprocessing.ForwardFill(backwards=True)),
        ]
    )

    # Test
    assert torch.isnan(pipeline.fit_transform(data)).sum() == 0
