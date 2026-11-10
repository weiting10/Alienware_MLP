#!/usr/bin/env python3
"""
MLP power-trace sweep: fully connected networks with random weights for every
combination of hidden layers x width, plus your trained MNIST MLP (784-250-100-10,
from Multilayer_Perceptron_with_PyTorch). One forward pass per trace.

Output: mlp_out/ (manifest.csv, traces/, raw/, plots/), same format as sweep.py.

    python sweep_mlp.py --match released_out --dry-run
    python sweep_mlp.py --match released_out --pilot
    python sweep_mlp.py --match released_out --no-plots
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_common as C


class WeitingMLP(nn.Module):
    """Same layers and names as Multilayer_Perceptron.py, so tut1-model.pt loads directly."""

    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        return self.output_fc(x)


class MLP(nn.Module):
    """input -> [Linear -> ReLU] x layers (all `width` wide) -> Linear(10)."""

    def __init__(self, input_dim, layers, width, classes=10):
        super().__init__()
        dims = [input_dim] + [width] * layers
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = nn.Linear(dims[-1], classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x)


def make_specs(args):
    specs = []
    for L in args.layers:                      # three nested loops -> one network each
        for W in args.width:
            specs.append(C.Spec("mlp", "mlp", f"mlp_L{L}_W{W}",
                                lambda L=L, W=W: MLP(args.input_dim, L, W), "vector",
                                f"{L} hidden layers x {W}", L, W, input_dim=args.input_dim))
    w = Path(args.mlp_weights)
    if w.exists():
        def trained():
            m = WeitingMLP(784, 10)
            m.load_state_dict(torch.load(w, map_location="cpu"))
            return m
        specs.append(C.Spec("mlp", "mlp", "weiting_mlp_784-250-100-10", trained, "vector",
                            "784-250-100-10 (your MNIST MLP)", 2, 250, "trained", input_dim=784))
    else:
        print(f"note: {w} not found, skipping your trained MLP (set --mlp-weights)")
    return specs


def main():
    p = C.common_args("Record GPU power traces of MLPs.", [1, 16, 64, 256, 1024], "mlp_out")
    a = p.add_argument_group("MLP sweep")
    a.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="hidden layers")
    a.add_argument("--width", nargs="+", type=int, default=[128, 256, 512, 1024, 2048],
                   help="neurons per hidden layer")
    a.add_argument("--input-dim", type=int, default=784, help="input features (784 = MNIST 28x28)")
    a.add_argument("--mlp-weights", default="../Multilayer_Perceptron_with_PyTorch/tut1-model.pt",
                   help="your trained MLP (added as an extra network if the file exists)")
    args = C.finish_args(p.parse_args(), "mlp")
    C.run(args, make_specs)


if __name__ == "__main__":
    main()