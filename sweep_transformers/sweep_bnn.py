#!/usr/bin/env python3
"""
BNN (binarized neural network) power-trace sweep, built from your
efficient-binary-neural-networks repo:
  - binarized MLP from the repo's BinaryLinear: hidden layers x width
  - LeNet5_binary (MNIST 28x28) with width_infl 1..5
  - VGGNet7_binary (CIFAR-10 32x32) with width_scale 0.25..2
  - the repo's trained LeNet5 and VGGNet7 binary checkpoints
One forward pass per trace.

Output: bnn_out/ (manifest.csv, traces/, raw/, plots/), same format as sweep.py.

    python sweep_bnn.py --match released_out --dry-run
    python sweep_bnn.py --match released_out --pilot
    python sweep_bnn.py --match released_out --no-plots
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_common as C


class BinaryMLP(nn.Module):
    """BinaryNet-style MLP from the repo's BinaryLinear: BinaryLinear -> BatchNorm -> hardtanh ->
    binarize, repeated; last BinaryLinear -> BatchNorm (same pattern as the repo's networks)."""

    def __init__(self, bm, input_dim, layers, width, classes=10):
        super().__init__()
        a = SimpleNamespace()
        dims = [input_dim] + [width] * layers
        self.bm = bm
        self.hidden = nn.ModuleList(bm.BinaryLinear(i, o, bias=False, args=a) for i, o in zip(dims, dims[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(o) for o in dims[1:])
        self.out = bm.BinaryLinear(dims[-1], classes, bias=False, args=a)
        self.out_bn = nn.BatchNorm1d(classes, affine=False)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer, bn in zip(self.hidden, self.bns):
            x = F.hardtanh(bn(layer(x)))
            x.data = self.bm.binarize(x.data)
        return self.out_bn(self.out(x))


def make_specs(args):
    repo, bm, lenet, vgg = C.import_bnn_repo(args.bnn_repo)
    specs = []
    for L in args.layers:
        for W in args.width:
            specs.append(C.Spec("bnn", "binary_mlp", f"bnn_mlp_L{L}_W{W}",
                                lambda L=L, W=W: BinaryMLP(bm, args.input_dim, L, W), "vector",
                                f"binarized MLP, {L} hidden layers x {W}", L, W, input_dim=args.input_dim))
    for k in args.lenet_widths:
        specs.append(C.Spec("bnn", "lenet5_binary", f"bnn_lenet5_w{k}",
                            lambda k=k: lenet.LeNet5_binary(C.bnn_args(repo, "LeNet5", width_infl=k)), "image",
                            f"LeNet5_binary, width_infl {k}", 4, 32 * k, fixed_size=(1, 28, 28)))
    for s in args.vgg_widths:
        specs.append(C.Spec("bnn", "vggnet7_binary", f"bnn_vggnet7_w{s:g}",
                            lambda s=s: vgg.VGGNet7_binary(C.bnn_args(repo, "VGGNet7", width_scale=s)), "image",
                            f"VGGNet7_binary, width_scale {s:g}", 9, int(128 * s), fixed_size=(3, 32, 32)))
    for net, size in (("LeNet5", (1, 28, 28)), ("VGGNet7", (3, 32, 32))):
        ckpt = repo / "pretrained" / net / "binary" / "results.pt"
        if not ckpt.exists():
            print(f"note: {ckpt} not found, skipping the trained {net}_binary")
            continue

        def trained(net=net, ckpt=ckpt):
            m = (lenet.LeNet5_binary if net == "LeNet5" else vgg.VGGNet7_binary)(C.bnn_args(repo, net))
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_best"])
            return m
        specs.append(C.Spec("bnn", f"{net.lower()}_binary", f"bnn_{net.lower()}_trained", trained,
                            "image", f"{net}_binary, trained checkpoint", "", "", "trained", fixed_size=size))
    return specs


def main():
    p = C.common_args("Record GPU power traces of binarized neural networks.",
                      [1, 16, 64, 256, 1024], "bnn_out")
    b = p.add_argument_group("BNN sweep (from your efficient-binary-neural-networks repo)")
    b.add_argument("--bnn-repo", default="../efficient-binary-neural-networks", help="path to the repo")
    b.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="binarized MLP hidden layers")
    b.add_argument("--width", nargs="+", type=int, default=[128, 256, 512, 1024, 2048],
                   help="binarized MLP neurons per layer")
    b.add_argument("--input-dim", type=int, default=784, help="binarized MLP input features")
    b.add_argument("--lenet-widths", nargs="+", type=int, default=[1, 2, 3, 4, 5], help="LeNet5_binary width_infl")
    b.add_argument("--vgg-widths", nargs="+", type=float, default=[0.25, 0.5, 1, 1.5, 2],
                   help="VGGNet7_binary width_scale")
    args = C.finish_args(p.parse_args(), "bnn")
    C.run(args, make_specs)


if __name__ == "__main__":
    main()