"""
problem.py
===============================
Methods for defining the problem.
"""
import numpy as np
import torch
from sklearn.base import TransformerMixin
from torch import nn

from tsbatteries.models import utils

from . import PadRaggedTensors

CRITERIONS = {
    "bce": nn.BCEWithLogitsLoss(),
    "continuous_bce": utils.TimeSeriesLossWrapper(nn.BCEWithLogitsLoss()),
    "ce": nn.CrossEntropyLoss(),
    "continuous_ce": utils.TimeSeriesLossWrapper(nn.BCEWithLogitsLoss()),
    "mse": nn.MSELoss(),
}


class LabelProcessor(TransformerMixin):
    """Builds a continuous problem from the labels.

    Todo:
        - Define for multiclass
    """

    def __init__(self, problem="online", lookback=0, lookforward=0, pad_sequence=False):
        assert problem in ["online", "oneshot", "regression"]
        self.problem = problem
        self.lookback = lookback
        self.lookforward = lookforward
        self.pad_sequence = pad_sequence

        self.split_labels = None
        self.stratify_index = None
        self.criterion = None

    def fit(self, labels, y=None):
        self._define_loss_criterion()
        if self.problem == "online":
            self.split_labels = torch.tensor([x.max() for x in labels])
            self.stratify_index = 0
        elif self.problem == "oneshot":
            self.split_labels = labels
            self.stratify = 0
        else:
            self.split_labels = torch.arange(len(labels))
            self.stratify = None
        return self

    def _define_loss_criterion(self):
        if self.problem == "online":
            self.criterion = CRITERIONS["continuous_bce"]
        elif self.problem == "oneshot":
            self.criterion = CRITERIONS["bce"]
        else:
            self.criterion = CRITERIONS["mse"]

    def transform(self, labels):
        if self.problem == "online":
            labels = _create_online_labels(
                labels, lookback=self.lookback, lookforward=self.lookforward
            )
        if self.pad_sequence:
            labels = PadRaggedTensors(fill_value=float("nan")).transform(labels)
        if self.problem == "online":
            # TODO: Whats going on here.
            labels = labels.unsqueeze(-1)
        return labels


def _create_online_labels(labels, lookback=0, lookforward=0):
    """ Loop over the ids and fill in labels according to the parameters. """
    for i in range(len(labels)):
        labels_ = labels[i]
        one_locations = np.argwhere(labels_ == 1).reshape(-1)
        if any(one_locations):
            for j in one_locations:
                labels_[j - lookback : j + lookforward] = 1
        labels[i] = labels_
    return labels
