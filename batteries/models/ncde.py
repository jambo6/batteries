from torch import nn
import torchcde


class NeuralCDE(nn.Module):
    """ Performs the Neural CDE training process over a batch of time series. """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 initial_dim=None,
                 hidden_hidden_dim=15,
                 num_layers=3,
                 apply_final_linear=True,
                 interpolation='cubic',
                 adjoint=False,
                 solver='rk4',
                 return_sequences=False):
        """
        Args:
            input_dim (int): The dimension of the path.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            initial_dim (int): The dimension of the initial values. If none, no initial network is specified.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
                net with the given density. Hidden and hidden hidden dims must be multiples of 32.
            adjoint (bool): Set True to use odeint_adjoint.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.initial_dim = initial_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.apply_final_linear = apply_final_linear
        self.interpolation = interpolation
        self.adjoint = adjoint
        self.solver = solver
        self.return_sequences = return_sequences

        # Initial to hidden
        if initial_dim is None:
            raise NotImplementedError()
        else:
            self.initial_linear = nn.Linear(initial_dim, hidden_dim)

        # The net that is applied to h_{t-1}
        self.func = _NCDEFunc(input_dim, hidden_dim, hidden_hidden_dim, num_layers)

        # Linear classifier to apply to final layer
        self.final_linear = nn.Linear(self.hidden_dim, self.output_dim) if apply_final_linear else lambda x: x

    def forward(self, inputs):
        assert len(inputs) == 2, 'Inputs must be a 2-tuple of (initial_values, coeffs)'
        initial, coeffs = inputs

        # Make lin int
        spline = torchcde.NaturalCubicSpline if self.interpolation == 'cubic' else torchcde.LinearInterpolation
        data = spline(coeffs)

        # Setup h0
        h0 = self.initial_linear(initial)

        # Perform the adjoint operation
        times = data.grid_points
        out = torchcde.cdeint(
            data, self.func, h0, times, adjoint=self.adjoint, method=self.solver
        )

        # Outputs
        outputs = self.final_linear(out[:, -1, :]) if not self.return_sequences else self.final_linear(out)

        # If rectilinear and return sequences, return every other value
        if all([self.return_sequences, self.interpolation == 'rectilinear']):
            outputs = outputs[:, ::2]

        return outputs


class _NCDEFunc(nn.Module):
    """The function applied to the hidden state in the NCDE model.

    This creates a simple RNN-like block to be used as the computation function f in:
        dh/dt = f(h) dX/dt
    """
    def __init__(self, input_dim, hidden_dim, hidden_hidden_dim=15, num_layers=1, density=0., rank=None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.sparsity = density
        self.rank = rank

        # Additional layers are just hidden to hidden with relu activation
        layers = [nn.Linear(hidden_dim, hidden_hidden_dim), nn.ReLU()]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_hidden_dim, hidden_hidden_dim), nn.ReLU()]

        # Add on final layer and Tanh and build net
        layers.append([nn.Linear(hidden_hidden_dim, hidden_dim * input_dim), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, t, h):
        return self.net(h).view(-1, self.hidden_dim, self.input_dim)
