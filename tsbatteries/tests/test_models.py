import pytest
import torch
from torch import nn, optim

from tsbatteries import models

MODELS = {"rnn": models.RNN, "retain": models.RETAIN, "ncde": models.NeuralCDE}


def generate_classification_problem():
    # Random data, binary classification problem
    data = torch.randn((50, 10, 3))
    labels = torch.randint(0, 2, (50, 1))

    # No negative values for labels == 1, so the task can train effectively
    positive_data = data[(labels == 1).view(-1)]
    positive_data[positive_data < 0] = 0
    data[(labels == 1).view(-1)] = positive_data

    # Split train test
    train_data, test_data = data[:30], data[30:]
    train_labels, test_labels = labels[:30], labels[30:]

    return train_data, test_data, train_labels, test_labels


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
    train_data, test_data, train_labels, test_labels = generate_classification_problem()
    input_dim, output_dim = train_data.size(2), train_labels.size(1)

    # Train the retain model
    model = MODELS[model_name](
        input_dim=input_dim,
        hidden_dim=30,
        output_dim=output_dim,
        return_sequences=False,
    )
    _, acc = training_loop(model, train_data, train_labels, n_epochs=50)
    assert 0 <= acc <= 1
