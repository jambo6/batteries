"""
problem.py
===============================
Methods for defining the problem.
"""
import numpy as np


def _assertions(data, label_index, data_columns, label_column, problem_type):
    # Assertions for `generate_labels`
    if problem_type == "online":
        assert data.dim() == 3
    elif problem_type == "oneshot":
        assert data.dim() == 2
    else:
        raise NotImplementedError(
            "Problem type must be 'online' or 'oneshot', recieved {}".format(
                problem_type
            )
        )

    # Ensure only one of label_index or label column is specified
    assert (label_index is None) ^ (label_column is None), (
        "Only one of `label_index` or `label_column` can be " "specified."
    )
    if label_column is not None:
        assert (
            label_column in data_columns
        ), "Cannot find `label_column` within `data_columns."
        assert (
            data_columns is not None
        ), "`data_columns` must be specified with `label_column`."
        col_size, data_size = len(data_columns), data.size(-1)
        assert col_size == data_size, (
            "`data_columns` must be the same length as the channel dim. Mismatched sizes "
            "{} and {}.".format(col_size, data_size)
        )


def _create_online_labels(labels, lookback=0, lookforward=0):
    """ Loop over the ids and fill in labels according to the parameters. """
    for i in range(len(labels)):
        labels_ = labels[i]
        for j in np.argwhere(labels_ == i):
            labels_[j - lookback : j + lookforward] = 1
        labels[i] = labels_
    return labels


def generate_labels(
    data,
    label_index=None,
    data_columns=None,
    label_column=None,
    problem_type=None,
):
    # Sanity
    _assertions(data, label_index, data_columns, label_column, problem_type)

    # Get the relevant labels and remove them from the data
    if label_column is not None:
        label_index = [
            i for i, column in enumerate(data_columns) if column == label_column
        ][0]
    data_indexes = [x for x in range(data.size(-1)) if x != label_index]
    labels = data[..., label_index]
    data = data[..., data_indexes]

    # Now reformat the labels in the case where the problem is online
    if problem_type == "online":
        labels = _create_online_labels(labels)

    return data, labels
