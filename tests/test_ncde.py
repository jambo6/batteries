import pytest
import torch
from torchcde import linear_interpolation_coeffs
from batteries.models import NeuralCDE
from .test_models import generate_classification_problem, training_loop


def create_ncde_problem(use_initial=True):
    # Simple problem
    train_data, test_data, train_labels, test_labels = generate_classification_problem()
    input_dim = train_data.size(-1)

    # Get initial
    train_initial = train_data[:, 0]
    test_initial = test_data[:, 0]

    # Make linear
    train_data = linear_interpolation_coeffs(train_data)
    test_data = linear_interpolation_coeffs(test_data)

    # Get some initial data
    initial_dim = None
    if use_initial:
        initial_dim = input_dim
        train_data = (train_initial, train_data)
        test_data = (test_initial, test_data)

    # Setup and NCDE
    output_dim = train_labels.size(1)
    hidden_dim = 15
    model = NeuralCDE(input_dim, hidden_dim, output_dim, initial_dim=initial_dim, interpolation='linear')

    return model, (train_data, train_labels), (test_data, test_labels)


def test_ncde_simple():
    # Test the model runs and gets a normal accuracy
    model, train_data, test_data = create_ncde_problem()
    _, acc = training_loop(model, *train_data, n_epochs=10)
    assert acc > 0.7

    # Check it runs with no initial values
    model, train_data, test_data = create_ncde_problem(use_initial=False)
    _, acc = training_loop(model, *train_data, n_epochs=10)
    assert acc > 0.7
    
    # Check no initial values raises an error if initial dim specified
    model.initial_dim = 10
    with pytest.raises(AssertionError):
        training_loop(model, *train_data, n_epochs=10)
