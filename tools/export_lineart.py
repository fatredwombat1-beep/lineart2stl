"""Build-time helper: download the "Informative Drawings" line-art generator
(the ControlNet "lineart" annotator weights) and export it to ONNX.

Runs in a throw-away Docker build stage that has PyTorch; the final image only
needs onnxruntime.

usage: python export_lineart.py OUT_DIR [--weights-dir DIR]
"""
import argparse
import os
import sys
import urllib.request

import torch
import torch.nn as nn

URLS = {
    "fine": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
    "bold": "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth",
}

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(f, f, 3), norm_layer(f), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(f, f, 3), norm_layer(f))

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super().__init__()
        self.model0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7),
                                    norm_layer(64), nn.ReLU(inplace=True))
        m1, fin, fout = [], 64, 128
        for _ in range(2):
            m1 += [nn.Conv2d(fin, fout, 3, stride=2, padding=1), norm_layer(fout), nn.ReLU(inplace=True)]
            fin, fout = fout, fout * 2
        self.model1 = nn.Sequential(*m1)
        self.model2 = nn.Sequential(*[ResidualBlock(fin) for _ in range(n_residual_blocks)])
        m3, fout = [], fin // 2
        for _ in range(2):
            m3 += [nn.ConvTranspose2d(fin, fout, 3, stride=2, padding=1, output_padding=1),
                   norm_layer(fout), nn.ReLU(inplace=True)]
            fin, fout = fout, fout // 2
        self.model3 = nn.Sequential(*m3)
        m4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            m4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*m4)

    def forward(self, x):
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


def export(model, path):
    dummy = torch.rand(1, 3, 256, 320)
    torch.onnx.export(model, (dummy,), path, input_names=["image"], output_names=["lines"],
                      dynamic_axes={"image": {2: "h", 3: "w"}, "lines": {2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--weights-dir", default=None, help="use local .pth files instead of downloading")
    ap.add_argument("--random", action="store_true", help="export untrained weights (testing only)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for style, url in URLS.items():
        model = Generator().eval()
        if not a.random:
            src = os.path.join(a.weights_dir, os.path.basename(url)) if a.weights_dir else None
            if not src or not os.path.exists(src):
                src = os.path.join(a.out_dir, os.path.basename(url))
                print("downloading", url, flush=True)
                urllib.request.urlretrieve(url, src)
            sd = torch.load(src, map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=True)      # fails loudly if the architecture is wrong
        out = os.path.join(a.out_dir, f"lineart_{style}.onnx")
        with torch.no_grad():
            export(model, out)
        print("wrote", out, os.path.getsize(out), "bytes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
