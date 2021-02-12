"""
problem.py
===============================
Methods for defining the problem.
"""
import numpy as np
from sklearn.base import TransformerMixin

from . import PadRaggedTensors


class LabelProcessor(TransformerMixin):
    """ Builds a continuous problem from the labels. """

    def __init__(self, problem="online", lookback=0, lookforward=0, pad_sequence=False):
        assert problem in ["online", "oneshot", "regression"]
        self.problem = problem
        self.lookback = lookback
        self.lookforward = lookforward
        self.pad_sequence = pad_sequence

    def fit(self, labels, y=None):
        return self

    def transform(self, labels):
        if self.problem == "online":
            labels = _create_online_labels(
                labels, lookback=self.lookback, lookforward=self.lookforward
            )
        if self.pad_sequence:
            labels = PadRaggedTensors(fill_value=float("nan")).transform(labels)
        return labels


def _create_online_labels(labels, lookback=0, lookforward=0):
    """ Loop over the ids and fill in labels according to the parameters. """
    for i in range(len(labels)):
        labels_ = labels[i]
        one_locations = np.argwhere(labels_ == 1)[0]
        if any(one_locations):
            for j in one_locations:
                labels_[j - lookback : j + lookforward] = 1
        labels[i] = labels_
    return labels
