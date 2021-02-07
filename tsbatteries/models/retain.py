import torch
from torch import nn


class RETAIN(nn.Module):
    """RETAIN model with time-series capabilities. """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        embedding_kwargs=None,
        return_sequences=False,
    ):
        super(RETAIN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_kwargs = embedding_kwargs
        self.return_sequences = return_sequences

        if self.return_sequences:
            raise NotImplementedError

        # Embedding dim if specified
        if embedding_kwargs is not None:
            self.setup_embedding_layer()

        # For extraction of most relevant series information
        self.series_sequential = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, batch_first=True),
            _ItemSelector(0),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(),
        )

        # For extraction of most relevant features
        self.features_sequential = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, batch_first=True),
            _ItemSelector(0),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

        # Output layer
        self.fc_output = nn.Linear(input_dim, output_dim)

    def _setup_embedding(self, embedding_kwargs):
        """ Setup an embedding if args are specified. """
        kwarg_keys = ["input_dropout", "embedding_dim", "embedding_dropout"]
        if any([embedding_kwargs.get(x) is None for x in kwarg_keys]):
            raise AssertionError(
                "Invalid keys for embedding_kwargs, must contain {}".format(kwarg_keys)
            )
        self.embedding = nn.Sequential(
            nn.Dropout(embedding_kwargs["input_dropout"]),
            nn.Linear(self.input_dim, embedding_kwargs["embedding_dim"]),
            nn.Dropout(embedding_kwargs["embedding_dropout"]),
        )
        self.input_dim = embedding_kwargs["embedding_dim"]

    @staticmethod
    def _generate_context(x, alpha, beta):
        """ Method for generating the context vector. """
        return torch.bmm(torch.transpose(alpha, 1, 2), beta * x).squeeze(1)

    def forward(self, x):
        if self.embedding_kwargs is not None:
            x = self.embedding(x)

        # Get series info
        alpha = self.series_sequential(x)

        # Get feature info
        beta = self.features_sequential(x)

        # Apply the final sum(alpha \times (beta dot x))
        context = self._generate_context(x, alpha, beta)

        # Linear layer and output
        output = self.fc_output(context)

        return output


class _ItemSelector(nn.Module):
    """ Used for extracting an item from a nn.Module that returns a tuple, such as an RNN. """

    def __init__(self, index):
        super(_ItemSelector, self).__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
