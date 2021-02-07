import pytest
import torch

from tsbatteries.models import NeuralCDE

from .test_models import generate_classification_problem, training_loop


def create_ncde_problem(static_dim=None, use_initial=True):
    # Simple problem
    train_data, test_data, train_labels, test_labels = generate_classification_problem()
    input_dim = train_data.size(-1)

    if static_dim is not None:
        static_train = torch.randn(train_data.size(0), static_dim)
        train_data = (static_train, train_data)
        static_test = torch.randn(test_data.size(0), static_dim)
        test_data = (static_test, test_data)

    # Setup and NCDE
    output_dim = train_labels.size(1)
    hidden_dim = 15
    model = NeuralCDE(
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=static_dim,
        interpolation="linear",
        use_initial=use_initial,
    )

    return model, (train_data, train_labels), (test_data, test_labels)


@pytest.mark.parametrize(
    "static_dim, use_initial",
    [(None, True), (None, False), (5, True), (5, False)],
)
def test_ncde_simple(static_dim, use_initial):
    # Test the model runs and gets a normal accuracy
    model, train_data, _ = create_ncde_problem(
        static_dim=static_dim, use_initial=use_initial
    )
    _, acc = training_loop(model, *train_data, n_epochs=10)
    assert acc > 0.7
