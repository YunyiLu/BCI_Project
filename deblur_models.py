"""
Stage 3: Deblurring Network — 模型定义
基于 Kim et al. 2021, Neural Computation — Baseline 5

论文对应: Section 4.5 Deblurring Network
- 使用 DeblurGANv2 的 ResNet Generator (6 个 ResNet block)
- 残差学习: output = input + model(input)
- 损失: L1 像素损失 + VGG-19 感知损失 (conv3_4 层)
- 无对抗损失 (论文明确排除)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple


# ============================================================
# ResNet Generator (from DeblurGANv2 / CycleGAN-pix2pix)
# 论文 Section 4.5: 6-block ResNet generator, grid search {1..8}
# ============================================================

class ResnetBlock(nn.Module):
    """标准 ResNet 残差块 (2×Conv3x3 + skip connection)"""

    def __init__(self, dim, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    DeblurGANv2 ResNet Generator
    论文参数: n_blocks=6, 无 Tanh 输出, 残差学习

    架构:
      Input (1, 64, 64)
        → ReflectionPad2d(3) → Conv7×7 (1→64) → IN → ReLU
        → Down×2: Conv3×3 stride=2 (64→128→256)
        → 6× ResnetBlock (256 channels)
        → Up×2: ConvTranspose2d stride=2 (256→128→64)
        → ReflectionPad2d(3) → Conv7×7 (64→1)
        → output = input + model(input)  [残差学习]
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=6,
                 use_dropout=False, learn_residual=True):
        """
        Args:
            input_nc:   输入通道数 (默认 1, 灰度)
            output_nc:  输出通道数 (默认 1, 灰度)
            ngf:        基础通道数 (默认 64)
            n_blocks:   ResNet 块数量 (论文 grid search 后选 6)
            use_dropout: 是否在 ResNet 块中使用 Dropout
            learn_residual: 是否学习残差 (论文默认 True)
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.learn_residual = learn_residual

        # --- 初始层: 7×7 Conv, stride=1 ---
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # --- 下采样 ×2: 64→32→16 ---
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                          stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # --- 瓶颈: n_blocks 个 ResNet 块 ---
        mult = 2 ** n_down  # 256
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, use_dropout=use_dropout)]

        # --- 上采样 ×2: 16→32→64 ---
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        # --- 输出层: 7×7 Conv, 无 Tanh ---
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        if self.learn_residual:
            out = x + out
        return out


# ============================================================
# VGG-19 感知损失
# 论文: L1 loss between features from conv3 of pretrained VGG-19
#      "before the corresponding pooling layer" → conv3_4 (relu3_4)
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 感知损失 (conv3_4 层特征)
    论文 Section 4.5: 使用预训练 VGG-19 的第三个卷积层特征
    公式: L_perceptual = ||VGG(img) - VGG(recon)||_1
    """

    def __init__(self, layer_name='relu3_4'):
        super(VGGPerceptualLoss, self).__init__()
        # 加载预训练 VGG-19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg_features = vgg.features

        # 提取到指定层 (relu3_4 是第 17 层, 索引从 0)
        self.layer_name = layer_name
        self.slice = self._get_layer_slice(vgg_features, layer_name)
        self.loss_net = nn.Sequential(*self.slice)

        # 冻结 VGG 参数
        for param in self.loss_net.parameters():
            param.requires_grad = False

        # VGG 均值/标准差归一化 (ImageNet 统计)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _get_layer_slice(features, target_layer):
        """截取 VGG features 到指定层"""
        slice_layers = []
        for name, module in features.named_children():
            slice_layers.append(module)
            if name == target_layer:
                break
        return slice_layers

    def to_rgb(self, x):
        """灰度 (1, H, W) → RGB (3, H, W): 三通道重复"""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, pred, target):
        """
        Args:
            pred:   预测图像 [B, 1, H, W]
            target: 真实图像 [B, 1, H, W]
        Returns:
            感知损失标量
        """
        pred = self.to_rgb(pred)
        target = self.to_rgb(target)

        # ImageNet 归一化
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_feat = self.loss_net(pred)
        target_feat = self.loss_net(target)

        return F.l1_loss(pred_feat, target_feat)


# ============================================================
# 组合损失 (论文: L1 + λ * VGG perceptual)
# ============================================================

class DeblurLoss(nn.Module):
    """
    论文损失函数:
      L_total = L1(pred, GT) + λ_perceptual * L_perceptual(pred, GT)
    无对抗损失 (论文明确排除)
    """

    def __init__(self, lambda_perceptual=0.5):
        """
        Args:
            lambda_perceptual: 感知损失权重 (论文未明确给出, 推荐 0.1~1.0)
        """
        super(DeblurLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptualLoss()
        self.lambda_perceptual = lambda_perceptual

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_perceptual = self.perceptual(pred, target)
        return loss_l1 + self.lambda_perceptual * loss_perceptual


# ============================================================
# 简化版 U-Net (备选)
# 如果 VGG 感知损失训练不稳定, 可回退到纯 L1 + U-Net
# ============================================================

class UNetDeblur(nn.Module):
    """
    简化 U-Net 去模糊网络 (备选)
    论文提到 "ongoing advances in image restoration networks"
    也提到 U-Net 风格的方法是可行的替代方案
    """

    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetDeblur, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, features)
        self.enc2 = self._conv_block(features, features * 2)
        self.enc3 = self._conv_block(features * 2, features * 4)
        self.enc4 = self._conv_block(features * 4, features * 8)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(features * 8, features * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.dec4 = self._conv_block(features * 16, features * 8)

        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.dec3 = self._conv_block(features * 8, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = self._conv_block(features * 4, features * 2)

        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = self._conv_block(features * 2, features)

        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.learn_residual = True

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        if self.learn_residual:
            out = x + out
        return out


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 快速测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("ResNet Generator (DeblurGANv2) 测试")
    print("=" * 60)

    model = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=6).to(device)
    print(f"参数量: {count_parameters(model):,}")
    print(f"输入: (1, 1, 64, 64)")

    x = torch.randn(1, 1, 64, 64).to(device)
    y = model(x)
    print(f"输出: {tuple(y.shape)}")
    print(f"残差学习: output = input + model(input) = {model.learn_residual}")

    print()
    print("=" * 60)
    print("VGG Perceptual Loss 测试")
    print("=" * 60)

    vgg_loss = VGGPerceptualLoss().to(device)
    a = torch.randn(2, 1, 64, 64).to(device)
    b = torch.randn(2, 1, 64, 64).to(device)
    loss = vgg_loss(a, b)
    print(f"VGG 感知损失: {loss.item():.6f}")

    print()
    print("=" * 60)
    print("U-Net 备选模型测试")
    print("=" * 60)

    unet = UNetDeblur(in_channels=1, out_channels=1).to(device)
    print(f"参数量: {count_parameters(unet):,}")
    y2 = unet(x)
    print(f"输出: {tuple(y2.shape)}")