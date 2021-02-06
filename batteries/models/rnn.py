from torch import nn

MODELS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RNN(nn.Module):
    """Standard RNN.

    Arguments:
        input_dim (int): The dimension of the path.
        hidden_dim (int): The dimension of the hidden state.
        output_dim (int): The dimension of the output.
        num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
            net with the given density. Hidden and hidden hidden dims must be multiples of 32.
        model_string (int): Any of ('rnn', 'gru', 'lstm')
        nonlinearity (str): One of ('tanh', 'relu').
        bias (bool): Whether to add a bias term.
        return_sequences (bool): If True will return the linear function on the final layer, else linear function on
            all layers.
        apply_final_linear (bool): Set False for no final linear layer to be applied to the hidden state.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        model_string="rnn",
        nonlinearity="tanh",
        bias=True,
        dropout=0,
        return_sequences=True,
        apply_final_linear=True,
    ):
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model_string = model_string
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.apply_final_linear = apply_final_linear

        model = MODELS.get(model_string)
        if model is None:
            raise NotImplementedError(
                "model_string must be one of {}, got {}".format(
                    MODELS.keys(), model_string
                )
            )
        self.rnn = model(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            dropout=dropout,
            batch_first=True,
        )
        self.total_hidden_size = num_layers * hidden_dim
        self.final_linear = (
            nn.Linear(self.total_hidden_size, output_dim)
            if self.apply_final_linear
            else lambda x: x
        )

    def forward(self, x):
        # Run the RNN
        h_full, _ = self.rnn(x)

        # Terminal output if classifcation else return all outputs
        outputs = (
            self.final_linear(h_full[:, -1, :])
            if not self.return_sequences
            else self.final_linear(h_full)
        )

        return outputs
