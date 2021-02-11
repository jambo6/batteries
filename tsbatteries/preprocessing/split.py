import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(
    tensors,
    val_frac=0.15,
    test_frac=0.15,
    stratify_idx=None,
    shuffle=True,
    seed=None,
    return_indices=False,
):
    """Train test split method for an arbitrary number of tensors.

    Given a list of tensor in the variable `tensors`, splits each randomly into (train_frac, val_frac, test_frac)
    proportions.

    Arguments:
        tensors (list): A list of torch tensors.
        val_frac (float): The fraction to use as validation data.
        test_frac (float): The fraction to use as test data.
        stratify_idx (int): The index of the `tensors` variable to use as stratification labels.
        shuffle (bool): Set True to shuffle first.
        return_indices (bool): Set True to return a 3-tuple of index locations (train_indices, val_indices,
            test_indices) rather than the split tensors themselves.
        seed (int): Random seed.

    Returns:
        A tuple containing three lists corresponding to the train/val/test split of `tensors` or the absolute index
            locations.
    """
    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Check all tensors have the same length
    num_samples = tensors[0].size(0)
    assert [t.size(0) == num_samples for t in tensors]

    # Add an indices tensor to get absolute index location
    tensors.append(np.arange(len(tensors[0])))

    # Stratification labels
    stratification_labels = None
    if stratify_idx is not None:
        stratification_labels = (
            tensors[stratify_idx] if tensors[stratify_idx].dim() <= 2 else None
        )

    # Get a train+val/test split followed by a train/val split.
    train_val_data, test_data = tensor_train_test_split(
        tensors, test_frac, stratify=stratification_labels, shuffle=shuffle
    )

    # Split out train and val
    if stratify_idx is not None:
        stratification_labels = train_val_data[stratify_idx]
    new_test_frac = val_frac / (1 - test_frac)
    train_data, val_data = tensor_train_test_split(
        train_val_data, new_test_frac, stratify=stratification_labels, shuffle=shuffle
    )

    # Return either of the indices or the full tensors
    tensors = [train_data, val_data, test_data]
    if return_indices:
        output = [x[-1] for x in tensors]
    else:
        output = [x[:-1] for x in tensors]

    return output


def tensor_train_test_split(tensors, test_frac, stratify=None, shuffle=True):
    """Splits a list of tensors into two parts according to the test_frac.

    Arguments:
        tensors (list): A list of tensors.
        test_frac (float): The fraction the test set.
        stratify (tensor): Stratification labels.
        shuffle (bool): Set True to shuffle first.

    Returns:
        Two lists, the first list contains the training tensors, the second the test tensors.
    """
    split_tensors = train_test_split(
        *tensors, stratify=stratify, shuffle=shuffle, test_size=test_frac
    )
    return split_tensors[0::2], split_tensors[1::2]
