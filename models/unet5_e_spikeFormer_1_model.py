import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul, BN1d, Conv1dT, MultiSpike
from spikingjelly.activation_based import layer

# v_th = 0.2
# alpha = 1 / (2 ** 0.5)


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, out_channels=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1,bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.body = nn.Sequential(
            MultiSpike(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride,bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.body(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = 2
        self.up = nn.Sequential(
            MultiSpike(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        input = F.interpolate(input, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.up(input)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            expansion_ratio=2,
            bias=False,
            kernel_size=7,
            padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpike()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike2 = MultiSpike()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels,
                      bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = MultiSpike()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.spike1(x)

        x = self.pwconv1(x)

        x = self.spike2(x)

        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = MultiSpike()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.spike2 = MultiSpike()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpike()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            lamda_ratio=1,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = MultiSpike()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = MultiSpike()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = MultiSpike()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
                                    nn.BatchNorm2d(int(dim * lamda_ratio)))

        self.v_spike = MultiSpike()

        self.attn_spike = MultiSpike()

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        ####
        self.matmul1 = SpikingMatmul('both')
        self.matmul2 = SpikingMatmul('r')

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = self.q_spike(q)
        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
                .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
                .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
                .reshape(B, N, self.num_heads, C_v // self.num_heads)
                .permute(0, 2, 1, 3)
                .contiguous()
        )

        # x = q @ k.transpose(-2, -1)
        # x = (x @ v) * (self.scale * 2)
        x = self.matmul1(q, k.transpose(-2, -1))
        x = self.matmul2(x, v) * (self.scale * 2)

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x


class MS_Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.conv = SepConv(dim=dim, kernel_size=3, padding=1)

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            lamda_ratio=4,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class SNNSIR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=32,
                 planes=[32, 64, 96, 128, 160],
                 num_heads=8,
                 mlp_ratio=4,
                 activation='LIF',
                 v_th=0.2,
                 alpha=(1 / (2 ** 0.5)),
                 bias=False,
                 decay_input=True,
                 v_reset=0.0,
                 *args,
                 **kwargs):
        super(SNNSIR, self).__init__()
        self.skip = ['patch_embed', 'project_out']
        self.patch_embed = OverlapPatchEmbed(in_channels, planes[0])

        self.encoder_level1 = MS_ConvBlock(planes[0], mlp_ratio)
        self.down1_2 = DownsampleLayer(planes[0], planes[1], 2)

        self.encoder_level2 = MS_ConvBlock(planes[1], mlp_ratio)
        self.down2_3 = DownsampleLayer(planes[1], planes[2], 2)

        self.encoder_level3 = MS_Block(planes[2], num_heads, mlp_ratio)
        self.down3_4 = DownsampleLayer(planes[2], planes[3], 2)
        #
        self.encoder_level4 = MS_Block(planes[3], num_heads, mlp_ratio)
        self.down4_5 = DownsampleLayer(planes[3], planes[4], 2)
        #
        self.scam = MS_Block(planes[4], num_heads, mlp_ratio)
        #
        self.up5_4 = UpsampleLayer(planes[4], planes[3], 1)
        self.decoder_level4 = MS_Block(planes[3], num_heads, mlp_ratio)
        #
        self.up4_3 = UpsampleLayer(planes[3], planes[2], 1)
        self.decoder_level3 = MS_Block(planes[2], num_heads, mlp_ratio)

        self.up3_2 = UpsampleLayer(planes[2], planes[1], 1)
        self.decoder_level2 = MS_ConvBlock(planes[1], mlp_ratio)

        self.up2_1 = UpsampleLayer(planes[1], planes[0], 1)
        self.decoder_level1 = MS_ConvBlock(planes[0], mlp_ratio)

        self.project_out = nn.Conv2d(planes[0], out_channels, 3, 1, 1, bias=False)


    def forward(self, x):
        shortcut = x.clone()

        ###deep feature extract###
        in_enc_level1 = self.patch_embed(x)

        out_enc_level1 = self.encoder_level1(in_enc_level1)
        in_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(in_enc_level2)
        in_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(in_enc_level3)
        in_enc_level4 = self.down3_4(out_enc_level3)

        out_enc_level4 = self.encoder_level4(in_enc_level4)
        in_enc_level5 = self.down4_5(out_enc_level4)

        out_dec_level5 = self.scam(in_enc_level5)

        in_dec_level4 = self.up5_4(out_dec_level5)
        in_dec_level4 = in_dec_level4 + out_enc_level4
        out_dec_level4 = self.decoder_level4(in_dec_level4)

        in_dec_level3 = self.up4_3(out_dec_level4)
        in_dec_level3 = in_dec_level3 + out_enc_level3
        out_dec_level3 = self.decoder_level3(in_dec_level3)

        in_dec_level2 = self.up3_2(out_dec_level3)
        in_dec_level2 = in_dec_level2 + out_enc_level2
        out_dec_level2 = self.decoder_level2(in_dec_level2)

        in_dec_level1 = self.up2_1(out_dec_level2)
        in_dec_level1 = in_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(in_dec_level1)

        out = self.project_out(out_dec_level1)
        out = out + shortcut

        return out

