import torch


def make_classification_problem(
    num_samples=50, length=10, input_dim=3, test_frac=0.4, static_dim=None
):
    # Random data, binary classification problem
    data = torch.randn((num_samples, length, input_dim))
    labels = torch.randint(0, 2, (num_samples, 1))

    # No negative values for labels == 1, so the task can train effectively
    positive_data = data[(labels == 1).view(-1)]
    positive_data[positive_data < 0] = 0
    data[(labels == 1).view(-1)] = positive_data

    # Split train test
    test_start = int(((1 - test_frac) * num_samples))
    train_data = [data[:test_start], labels[:test_start]]
    test_data = [data[test_start:], labels[test_start:]]

    # Add static if set
    if static_dim is not None:
        static_data = torch.randn(num_samples, static_dim)
        train_data = [static_data[:test_start]] + train_data
        test_data = [static_data[test_start:]] + test_data

    return train_data, test_data
