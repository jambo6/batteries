import torch
from torch import nn
import torchcde

SPLINES = {
    'cubic': torchcde.NaturalCubicSpline,
    'linear': torchcde.LinearInterpolation,
    'rectilinear': torchcde.LinearInterpolation,
}


class NeuralCDE(nn.Module):
    """ Performs the Neural CDE training process over a batch of time series. """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        initial_dim=None,
        hidden_hidden_dim=15,
        num_layers=3,
        apply_final_linear=True,
        interpolation="cubic",
        adjoint=False,
        solver="rk4",
        return_sequences=False,
    ):
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

        # Interpolation function
        self.spline = SPLINES.get(self.interpolation)
        if self.spline is None:
            spline_keys = SPLINES.keys()
            raise NotImplementedError("Allowed interpolation schemes {}, given {}".format(spline_keys, interpolation))

        # Initial to hidden
        if initial_dim is not None:
            self.initial_linear = nn.Linear(initial_dim, hidden_dim)

        # The net that is applied to h_{t-1}
        self.func = _NCDEFunc(input_dim, hidden_dim, hidden_hidden_dim, num_layers)

        # Linear classifier to apply to final layer
        self.final_linear = (
            nn.Linear(self.hidden_dim, self.output_dim)
            if apply_final_linear
            else lambda x: x
        )

    def setup_h0(self, inputs):
        """ Sets up h0 according to given input. """
        if self.initial_dim is not None:
            assert len(inputs) == 2, "Inputs must be a 2-tuple of (initial_values, coeffs)"
            initial, coeffs = inputs
            h0 = self.initial_linear(initial)
        else:
            coeffs = inputs
            batch_dim, length_dim, hidden_dim = coeffs.size(0), coeffs.size(1), self.hidden_dim
            h0 = torch.autograd.Variable(torch.zeros(batch_dim, length_dim, hidden_dim)).to(coeffs.device)
        return coeffs, h0

    def forward(self, inputs):
        # Handle h0 and inputs
        coeffs, h0 = self.setup_h0(inputs)

        # Make lin int
        data = self.spline(coeffs)

        # Perform the adjoint operation
        out = torchcde.cdeint(
            data, self.func, h0, data.grid_points, adjoint=self.adjoint, method=self.solver
        )

        # Outputs
        outputs = (
            self.final_linear(out[:, -1, :])
            if not self.return_sequences
            else self.final_linear(out)
        )

        # If rectilinear and return sequences, return every other value
        if all([self.return_sequences, self.interpolation == "rectilinear"]):
            outputs = outputs[:, ::2]

        return outputs


class _NCDEFunc(nn.Module):
    """The function applied to the hidden state in the NCDE model.

    This creates a simple RNN-like block to be used as the computation function f in:
        dh/dt = f(h) dX/dt
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_hidden_dim=15,
        num_layers=1,
        density=0.0,
        rank=None,
    ):
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
