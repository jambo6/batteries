from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader, TensorDataset

from tsbatteries.models.utils import StaticTemporalTensorDataset
from tsbatteries.preprocessing.split import train_val_test_split


class PipelineDataset:
    """ Class that holds static, temporal and label data that can be fed into the PipelineCompiler. """

    def __init__(self, static_data=None, temporal_data=None, labels=None):
        """
        Args:
            static_data (tensor): Static data of shape
            temporal_data (tensor or list of tensors):
            labels (tensor):
        """
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.labels = labels

    def __len__(self):
        return len(self.temporal_data)


class PipelineCompiler(TransformerMixin):
    """Combines preprocessing pipelines and data splitting.

    Given a PipelineDataset, the fit method will first split the data into train/val/test, then apply the pipelines to
    the relevant data on the training set only. The split indices are saved as attributes that will be used in the
    transform to return the transforms on each split.

    Attributes:
        train_indices (list): The index locations of the training data on the fitted dataset.
        val_indices (list): As train_indices but validation.
        test_indices (list): As train_indices but test.
        indicies (list): A list containing [train_indices, val_indices, test_indices]. Useful for easy iteration.
    """

    def __init__(
        self, static_pipeline, temporal_pipeline, label_pipeline, batch_size=None
    ):
        """
        Args:
            static_pipeline (sklearn pipeline or None): The pipeline to be applied to the static data. Leave as None
                if no static data is being used.
            temporal_pipeline (sklearn pipeline): The pipeline to be applied to the temporal data.
            label_pipeline (sklearn pipeline): The pipeline to be applied to the labels.
            batch_size (int): Set this only if you wish the output of the transform method to be three dataloaders
                rather than tensor lists.
        """
        self.static_pipeline = static_pipeline
        self.temporal_pipeline = temporal_pipeline
        self.label_pipeline = label_pipeline
        self.batch_size = batch_size

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    @property
    def indices(self):
        return [self.train_indices, self.val_indices, self.test_indices]

    def fit(self, dataset):
        """ Applies the train/val/test split and then fits the pipeline to the training data. """
        # Split
        self.train_indices, self.val_indices, self.test_indices = train_val_test_split(
            [dataset.labels], return_indices=True, stratify_idx=0
        )

        # Apply the pipelines to the training data
        if dataset.static_data is not None:
            self.static_pipeline.fit(dataset.static_data[self.train_indices])
        self.temporal_pipeline.fit(dataset.temporal_data[self.train_indices])

        return self

    def _apply_pipeline_to_indices(self, dataset, indices):
        # Get temporal and labels
        temporal_data = self.temporal_pipeline.transform(dataset.temporal_data[indices])
        labels = dataset.labels[indices]
        tensors = [temporal_data, labels]

        # Finally add static if we have static data
        if dataset.static_data is not None:
            static_data = self.static_pipeline.transform(dataset.static_data[indices])
            tensors = [static_data] + tensors

        return tensors

    def _to_dataloader(self, tensors):
        # Convert into a dataloader
        dataset_class = (
            StaticTemporalTensorDataset if len(tensors) == 3 else TensorDataset
        )
        dataset = dataset_class(*tensors)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def transform(self, dataset):
        """Applies the pipeline transformation methods to the train/val/test data in turn.

        Returns:
            If batch size is not set, returns a list of lists of tensors with the inner lists containing [static,
                temporal, labels] for train/val/test respectively.
            If batch size is set returns a list of dataloaders that load from a TensorDataset(temporal, labels) if no
                static data is set, else from StaticTemporalTensorDataset.
        """
        assert all(
            [x is not None for x in self.indices]
        ), "No split indices found, check this has been fitted."

        # Transform the pipelines to each split
        outputs = [
            self._apply_pipeline_to_indices(dataset, indices)
            for indices in self.indices
        ]

        # Convert to dataloaders if specified
        if self.batch_size is not None:
            outputs = [self._to_dataloader(tensors) for tensors in outputs]

        return outputs
