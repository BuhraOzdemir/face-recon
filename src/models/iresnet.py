"""
IResNet50 — InsightFace ArcFace backbone (PyTorch native).

Kaynak: https://github.com/deepinsight/insightface (MIT License)
Kullanım:
    model = iresnet50(pretrained_path='/path/to/backbone.pth')
    model.eval()
    emb = model(face_112x112_tensor)  # (B, 512) L2-normalized

Not: Bu model yalnızca eğitim sırasında identity loss için kullanılır.
     Telefona yüklenmez.
"""

import torch
import torch.nn as nn
from pathlib import Path


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample=None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1):
        super().__init__()
        self.bn1      = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1    = conv3x3(inplanes, planes)
        self.bn2      = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu    = nn.PReLU(planes)
        self.conv2    = conv3x3(planes, planes, stride)
        self.bn3      = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride   = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """
    IResNet backbone — InsightFace ArcFace face recognition modeli.

    Giriş : (B, 3, 112, 112)  — normalized [-1, 1] veya [0, 1]
    Çıkış : (B, 512)           — L2-normalized embedding
    """

    def __init__(self, block, layers, num_features: int = 512, dropout: float = 0.0):
        super().__init__()
        self.inplanes = 64

        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes, eps=1e-5)
        self.prelu  = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2     = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        # 112×112 giriş → 4 adet stride-2 layer → 7×7 feature map
        self.fc       = nn.Linear(512 * block.expansion * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad_(False)

        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet50(pretrained_path: str = None, dropout: float = 0.0) -> IResNet:
    """
    IResNet50 yükle.

    Args:
        pretrained_path: .pth dosyası yolu. None ise rastgele ağırlıklar.
        dropout: Dropout oranı (varsayılan 0.0).

    Returns:
        IResNet modeli (eval modu, frozen).
    """
    model = IResNet(IBasicBlock, [3, 4, 14, 3], dropout=dropout)

    if pretrained_path is not None:
        path = Path(pretrained_path)
        if not path.exists():
            raise FileNotFoundError(
                f"IResNet50 ağırlık dosyası bulunamadı: {pretrained_path}\n"
                f"İndirmek için train_kaggle.ipynb'deki 'Ağırlık İndir' hücresini çalıştır."
            )
        state = torch.load(pretrained_path, map_location="cpu")
        # InsightFace checkpoint bazen 'state_dict' anahtarı içerir
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[IResNet50] Eksik anahtar sayısı: {len(missing)}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


if __name__ == "__main__":
    model = iresnet50()
    dummy = torch.randn(2, 3, 112, 112)
    out   = model(dummy)
    print(f"IResNet50 çıkış: {out.shape}")  # (2, 512)
