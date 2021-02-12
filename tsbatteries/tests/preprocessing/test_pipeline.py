import numpy as np
import pytest
import torch

from tsbatteries import preprocessing
from tsbatteries.preprocessing.pipeline import (PipelineCompiler,
                                                PipelineDataset)
from tsbatteries.tests.helpers import make_classification_problem


@pytest.fixture()
def static_pipeline():
    pipeline = preprocessing.Pipeline(
        [
            ("negative_impute", preprocessing.NegativeFilter()),
            ("stdsc", preprocessing.TensorScaler(method="mms")),
        ]
    )
    return pipeline


@pytest.fixture()
def temporal_pipeline():
    pipeline = preprocessing.Pipeline(
        [
            ("pad", preprocessing.PadRaggedTensors()),
            ("negative_impute", preprocessing.NegativeFilter()),
            ("stdsc", preprocessing.TensorScaler(method="stdsc")),
            ("interpolation", preprocessing.Interpolation(method="linear")),
            ("backfill", preprocessing.ForwardFill(backwards=True)),
        ]
    )
    return pipeline


@pytest.fixture()
def label_pipeline():
    pipeline = preprocessing.LabelProcessor(problem="oneshot")
    return pipeline


def test_pipeline(temporal_pipeline):
    # Create random tensor data
    data = [torch.randn(5, 2) for _ in range(5)]
    assert torch.isnan(temporal_pipeline.fit_transform(data)).sum() == 0


@pytest.mark.parametrize(
    "static_dim, batch_size",
    [
        (None, None),
        (None, 32),
        (5, None),
        (5, 32),
    ],
)
def test_preprocessing_pipeline(
    static_pipeline, temporal_pipeline, label_pipeline, static_dim, batch_size
):
    # Check the preprocessing pipeline can be run effectively with and without static data
    train_data, _ = make_classification_problem(static_dim=5)
    static_data, temporal_data, labels = train_data
    temporal_data = np.array([x for x in temporal_data], dtype=object)
    if static_dim is None:
        static_data = None
        static_pipeline = None

    # Check fit transform works with full data
    dataset = PipelineDataset(static_data, temporal_data, labels)
    main_pipeline = PipelineCompiler(
        static_pipeline, temporal_pipeline, label_pipeline, batch_size=batch_size
    )
    output = main_pipeline.fit_transform(dataset)

    # Check fit transform works with and without static
    assert len(output) == 3
