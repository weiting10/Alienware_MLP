#!/usr/bin/env python3
"""
CNN power-trace sweep: torchvision CNN families (VGG, ResNet, MobileNetV2/V3,
EfficientNet, DenseNet) over batch size x image size, plus the full-precision
LeNet5/VGGNet7 from your efficient-binary-neural-networks repo ("rwth" family).
Random weights by default (no downloads); --pretrained loads ImageNet weights.
One forward pass per trace.

Output: cnn_out/ (manifest.csv, traces/, raw/, plots/), same format as sweep.py.

    python sweep_cnn.py --match released_out --dry-run
    python sweep_cnn.py --match released_out --pilot
    python sweep_cnn.py --match released_out --no-plots
"""

import sys

import nn_common as C

# Each family's variants, smallest to largest: (torchvision function, extra options)
FAMILIES = {
    "vgg": [("vgg11", {}), ("vgg11_bn", {}), ("vgg13", {}), ("vgg13_bn", {}),
            ("vgg16", {}), ("vgg16_bn", {}), ("vgg19", {}), ("vgg19_bn", {})],
    "resnet": [("resnet18", {}), ("resnet34", {}), ("resnet50", {}), ("resnet101", {}), ("resnet152", {})],
    "mobilenet_v2": [("mobilenet_v2", {"width_mult": m}) for m in (0.5, 0.75, 1.0, 1.4)],
    "mobilenet_v3": [("mobilenet_v3_small", {}), ("mobilenet_v3_large", {})],
    "efficientnet": [(f"efficientnet_b{i}", {}) for i in range(5)],
    "densenet": [("densenet121", {}), ("densenet169", {}), ("densenet201", {}), ("densenet161", {})],
}


def torchvision_model(name, pretrained, **kw):
    import torchvision.models as M
    return getattr(M, name)(weights="DEFAULT" if pretrained else None, **kw)


def make_specs(args):
    unknown = [f for f in args.families if f not in FAMILIES and f != "rwth"]
    if unknown:
        sys.exit(f"unknown CNN families {unknown}; choose from {list(FAMILIES) + ['rwth']}")
    specs = []
    for fam in [f for f in args.families if f in FAMILIES]:
        for name, kw in FAMILIES[fam]:
            mid = name + (f"_x{kw['width_mult']:g}" if kw else "")
            specs.append(C.Spec("cnn", fam, mid,
                                lambda name=name, kw=kw: torchvision_model(name, args.pretrained, **kw),
                                "image", mid, weights="trained" if args.pretrained else "random"))
    if "rwth" in args.families:
        repo, bm, lenet, vgg = C.import_bnn_repo(args.bnn_repo)
        specs.append(C.Spec("cnn", "rwth", "lenet5_float", lambda: lenet.LeNet5(C.bnn_args(repo, "LeNet5", False)),
                            "image", "LeNet5 (full precision)", 4, 32, fixed_size=(1, 28, 28)))
        specs.append(C.Spec("cnn", "rwth", "vggnet7_float", lambda: vgg.VGGNet7("cifar10"),
                            "image", "VGGNet7 (full precision)", 9, 128, fixed_size=(3, 32, 32)))
    return specs


def main():
    p = C.common_args("Record GPU power traces of CNNs.", [1, 4, 16, 32, 64], "cnn_out")
    c = p.add_argument_group("CNN sweep")
    c.add_argument("--families", nargs="+", default=list(FAMILIES) + ["rwth"],
                   help="rwth = full-precision LeNet5/VGGNet7 from your BNN repo")
    c.add_argument("--image-size", nargs="+", type=int, default=[64, 112, 160, 224, 288],
                   help="square input image sizes (LeNet5/VGGNet7 use their fixed size)")
    c.add_argument("--pretrained", action="store_true",
                   help="load ImageNet-trained weights (downloads them) instead of random weights")
    c.add_argument("--bnn-repo", default="../efficient-binary-neural-networks",
                   help="path to your BNN repo (for the rwth family)")
    args = C.finish_args(p.parse_args(), "cnn")
    C.run(args, make_specs)


if __name__ == "__main__":
    main()