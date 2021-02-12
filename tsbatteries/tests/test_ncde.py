import pytest

from tsbatteries.models import NeuralCDE

from .helpers import make_classification_problem
from .test_models import training_loop


def create_ncde_problem(static_dim=None, use_initial=True):
    # Simple problem
    train_data, test_data = make_classification_problem(static_dim=static_dim)

    # Get sizes
    input_dim, output_dim = [t.size(-1) for t in train_data][-2:]

    # Setup and NCDE
    hidden_dim = 15
    model = NeuralCDE(
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=static_dim,
        interpolation="linear",
        use_initial=use_initial,
    )

    # Function to convert to ncde format data
    def to_ncde(tensors):
        if static_dim is not None:
            tensors = (tensors[0], tensors[1]), tensors[2]
        return tensors

    return model, to_ncde(train_data), to_ncde(test_data)


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
    assert 0 <= acc <= 1
