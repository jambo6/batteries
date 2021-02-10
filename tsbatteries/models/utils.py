import numpy as np


class StaticTemporalTensorDataset:
    """ Outputs a tuple of (static, temporal) and labels at the specified indices. """

    def __init__(self, static_data, temporal_data, labels):
        super(StaticTemporalTensorDataset, self).__init__()
        assert len(static_data) == len(temporal_data) == len(labels)
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.labels = labels

    def __getitem__(self, index):
        return (self.static_data[index], self.temporal_data[index]), self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_num_params(model):
    """Returns the number of trainable parameters in a pytorch model.

    Arguments:
        model (nn.Module): PyTorch model.

    Returns:
        An integer denoting the number of trainable parameters in the model.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
