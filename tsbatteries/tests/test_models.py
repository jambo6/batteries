import pytest
import torch
from torch import nn, optim

from tsbatteries import models
from tsbatteries.models import utils
from tsbatteries.tests.helpers import make_classification_problem

MODELS = {"rnn": models.RNN, "retain": models.RETAIN, "ncde": models.NeuralCDE}


def training_loop(model, data, labels, n_epochs=5):
    labels = labels.to(torch.float)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
    # Evaluate on train
    preds = torch.sigmoid(model(data))
    acc = ((preds - labels) < 0.5).sum() / len(preds)
    return preds, acc


@pytest.mark.parametrize(
    "model_name",
    ["rnn", "retain"],
)
def test_rnn_models(model_name):
    # Test full accuracy on an easy classification problem
    # Note that this additionally tests for standardised inputs
    # Load a basic classification problem
    train_data, _ = make_classification_problem()
    data, labels = train_data
    input_dim, output_dim = data.size(2), labels.size(1)

    # Train the retain model
    model = MODELS[model_name](
        input_dim=input_dim,
        hidden_dim=30,
        output_dim=output_dim,
        return_sequences=False,
    )
    _, acc = training_loop(model, data, labels, n_epochs=50)
    assert 0 <= acc <= 1


def test_tune_number_of_parameters():
    # Load a basic classification problem
    train_data, _ = make_classification_problem(static_dim=None)
    data, labels = train_data
    input_dim, output_dim = data.size(2), labels.size(1)

    # Create a model builder
    def model_builder(x):
        return MODELS["rnn"](
            input_dim=input_dim,
            hidden_dim=x,
            output_dim=output_dim,
            return_sequences=False,
        )

    # Check works
    for num_params in [1000, 50000, 100000]:
        model = utils.tune_number_of_parameters(model_builder, num_params)
        assert 0.9 * num_params < utils.get_num_params(model) < 1.1 * num_params
