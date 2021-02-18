import numpy as np
import torch
from torch import nn


class SupervisedLearningDataset:
    """Dataset for a collection of input tensors and corresponding labels.

    Given a set of inputs (inputs_1, ..., inputs_n) with corresponding labels, the __getitem__ method returns
        [inputs_1[item], ..., inputs_n[item]], labels[item]

    In general model forward methods accept a single input. If this input is a tuple, additional logic takes place
    within the class. This class allows for multiple inputs to be used whilst still resulting in a single input to
    the model forward method. This makes things easier when using pre-existing tools (e.g. ignite) to handle batch
    unpacking since no modifications to the provided factory functions are required.
    """

    def __init__(self, tensors, labels):
        super(SupervisedLearningDataset, self).__init__()

        self._assert(tensors, labels)

        self.tensors = tensors
        self.labels = labels

    def _assert(self, tensors, labels):
        assert any(
            [isinstance(tensors, x) for x in [list, torch.Tensor]]
        ), "tensors must be a list or tensor"
        if isinstance(tensors, list):
            self.is_list = True
            assert [len(x) == len(labels) for x in tensors]
        else:
            self.is_list = False
            assert len(tensors) == len(labels)

    def __getitem__(self, index):
        if self.is_list:
            return [t[index] for t in self.tensors], self.labels[index]
        else:
            return self.tensors[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class TimeSeriesLossWrapper(nn.Module):
    """Applies a given loss function along the length of the outputs removing any nans in the given labels.

    Given a PyTorch loss function and target labels of shape [N, L, C] (as opposed to [N, C]), this removes any
    values where labels are nan (corresponding to a finished time series) and computes the loss against the labels
    and predictions in the remaining non nan locations.
    """

    def __init__(self, criterion):
        """
        Args:
            criterion (nn.Module): A pytorch loss function.
        """
        super().__init__()

        assert isinstance(criterion, nn.Module)
        self.criterion = criterion

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        return self.criterion(preds[mask], labels[mask])


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


def tune_number_of_parameters(model_builder, desired_num_params):
    """Tunes a model parameter to make the total model params as close to desired_num_params as possible.

    Given a function that accepts an integer argument and outputs an initialised nn.Module, increments the argument
    until the number of parameters in the output model is as close to `desired_num_params` as possible.

    Args:
        model_builder (function): A function that accepts an integer argument and outputs a model.
        desired_num_params (int): The number of parameters wanted in the output model.

    Returns:
        An initialised model with number of params close to `desired_num_params`.
    """
    min_params = get_num_params(model_builder(1))
    if min_params > desired_num_params:
        raise ValueError(
            "Minimum number of model params ({}) is more than desired ({})".format(
                min_params, desired_num_params
            )
        )

    num_params = 0
    hidden_dim = 1
    while num_params < desired_num_params:
        model = model_builder(hidden_dim)
        num_params = get_num_params(model)
        hidden_dim += 1

    # Revert back one if it is closer
    if hidden_dim - 2 > 0:
        prev_build = model_builder(hidden_dim - 2)
        if (
            desired_num_params - get_num_params(prev_build)
            < desired_num_params - num_params
        ):
            model = prev_build

    return model
