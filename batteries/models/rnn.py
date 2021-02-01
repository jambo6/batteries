from torch import nn


class RNN(nn.Module):
    """ Standard RNN. """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, nonlinearity='tanh', bias=True,
                 dropout=0, return_sequences=True, apply_final_linear=True):
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.apply_final_linear = apply_final_linear

        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)
        self.total_hidden_size = num_layers * hidden_dim
        self.final_linear = nn.Linear(self.total_hidden_size, output_dim) if self.apply_final_linear else lambda x: x

    def forward(self, x):
        # Run the RNN
        h_full, _ = self.rnn(x)

        # Terminal output if classifcation else return all outputs
        outputs = self.final_linear(h_full[:, -1, :]) if not self.return_sequences else self.final_linear(h_full)

        return outputs


class GRU(nn.Module):
    """ Standard GRU. """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, bias=True, dropout=0,
                 return_sequences=True):
        super(GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.return_sequences = return_sequences

        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bias=bias,
                          dropout=dropout,
                          batch_first=True)
        self.total_hidden_size = num_layers * hidden_dim
        self.final_linear = nn.Linear(self.total_hidden_size, output_dim)

    def forward(self, x):
        # Run the RNN
        h_full, h_final = self.gru(x)

        # Terminal output if classification else return all outputs
        outputs = self.final_linear(h_final.squeeze(0)) if not self.return_sequences else self.linear(h_full)

        return outputs
