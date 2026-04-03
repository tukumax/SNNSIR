import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul, BN1d, Conv1dT, LIFSigmoid
from spikingjelly.activation_based import layer

# v_th = 0.2
# alpha = 1 / (2 ** 0.5)


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, out_channels=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = Conv3x3(in_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.proj(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super().__init__()
        # self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        # self.norm = BN(num_features=out_channels, v_th=v_th, alpha=alpha)
        # self.activation = activation(v_threshold=v_th)
        self.body = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(in_channels, out_channels, stride=stride),
            BN(num_features=out_channels, v_th=v_th, alpha=alpha)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.activation(x)
        # x = self.conv(x)
        # x = self.norm(x)
        x = self.body(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = 2
        self.up = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(in_channels, out_channels, stride=stride),
            BN(num_features=out_channels, v_th=v_th, alpha=alpha),
        )

    def forward(self, input):
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3, activation=LIF, v_th=0.2,
        decay_input=True, v_reset=0.0
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.lif3 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn2(self.pwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,activation=LIF, v_th=0.2,
        decay_input=True, v_reset=0.0
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim, activation=activation, v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.lif2 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None,activation=LIF, v_th=0.2,
        decay_input=True, v_reset=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,activation=LIF, v_th=0.2,
        decay_input=True, v_reset=0.0
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)

        self.k_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)

        self.v_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)

        self.attn_lif = activation(v_threshold=v_th/2, decay_input=decay_input, v_reset=v_reset)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(dim)
        )
        ####
        self.matmul1 = SpikingMatmul('both')
        self.matmul2 = SpikingMatmul('l')

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # x = k.transpose(-2, -1) @ v
        # x = (q @ x) * self.scale
        x = self.matmul1(k.transpose(-2, -1), v)
        x = self.matmul2(q, x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,activation=LIF, v_th=0.2,
        decay_input=True, v_reset=0.0
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            activation=activation, v_th=v_th,
            decay_input=decay_input, v_reset=v_reset
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim,activation=activation, v_th=v_th, decay_input=decay_input, v_reset=v_reset)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SNNSIR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=48,
                 T=4,
                 planes=[32, 64, 96, 128, 160],
                 num_heads=8,
                 mlp_ratio=4,
                 activation='LIF',
                 v_th=0.2,
                 alpha=(1 / (2 ** 0.5)),
                 sr_scale=4,
                 decay_input=True,
                 v_reset=0.0,
                 *args,
                 **kwargs):
        super(SNNSIR, self).__init__()
        # 配置读取的原因
        if activation == 'LIF':
            activation = LIF
        elif activation == 'LIFSigmoid':
            activation = LIFSigmoid
        else:
            activation = PLIF

        self.T = T
        self.skip = ['patch_embed', 'project_out','left_scale_up_0_1','left_scale_up_1_1']
        self.patch_embed = OverlapPatchEmbed(in_channels, planes[0])

        self.encoder_level1 = MS_ConvBlock(planes[0], mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        # self.down1_2 = DownsampleLayer(planes[0], planes[1], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level2 = MS_ConvBlock(planes[1], mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        # self.down2_3 = DownsampleLayer(planes[1], planes[2], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level3 = MS_Block(planes[2], num_heads, mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        # self.down3_4 = DownsampleLayer(planes[2], planes[3], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        #
        self.encoder_level4 = MS_Block(planes[3], num_heads, mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        # self.down4_5 = DownsampleLayer(planes[3], planes[4], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        #
        self.scam = MS_Block(planes[4], num_heads, mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        #
        # self.up5_4 = UpsampleLayer(planes[4], planes[3], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level4 = MS_Block(planes[3], num_heads, mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)
        #
        # self.up4_3 = UpsampleLayer(planes[3], planes[2], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level3 = MS_Block(planes[2], num_heads, mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)

        # self.up3_2 = UpsampleLayer(planes[2], planes[1], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level2 = MS_ConvBlock(planes[1], mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)

        # self.up2_1 = UpsampleLayer(planes[1], planes[0], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level1 = MS_ConvBlock(planes[0], mlp_ratio, activation,v_th=v_th, decay_input=decay_input, v_reset=v_reset)

        self.project_out = nn.Conv2d(planes[0], out_channels, 3, 1, 1, bias=False)

        #
        self.sr_scale = sr_scale
        self.left_scale_up_0_1 = nn.Conv2d(out_channels, out_channels * (sr_scale ** 2), kernel_size=3, stride=1,
                                           padding=1, bias=False)
        self.left_scale_up_1_1 = nn.PixelShuffle(sr_scale)


    def forward(self, x):
        # shortcut = x.clone()
        shortcut = F.interpolate(x, scale_factor=self.sr_scale, mode='bicubic', align_corners=False)

        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        ###deep feature extract###
        out = self.patch_embed(x)

        out = self.encoder_level1(out)
        # in_enc_level2 = self.down1_2(out_enc_level1)

        out = self.encoder_level2(out)
        # in_enc_level3 = self.down2_3(out_enc_level2)

        out = self.encoder_level3(out)
        # in_enc_level4 = self.down3_4(out_enc_level3)

        out = self.encoder_level4(out)
        # in_enc_level5 = self.down4_5(out_enc_level4)

        out = self.scam(out)

        # in_dec_level4 = self.up5_4(out_dec_level5)
        # in_dec_level4 = in_dec_level4 + out_enc_level4
        out = self.decoder_level4(out)

        # in_dec_level3 = self.up4_3(out_dec_level4)
        # in_dec_level3 = in_dec_level3 + out_enc_level3
        out = self.decoder_level3(out)

        # in_dec_level2 = self.up3_2(out_dec_level3)
        # in_dec_level2 = in_dec_level2 + out_enc_level2
        out = self.decoder_level2(out)

        # in_dec_level1 = self.up2_1(out_dec_level2)
        # in_dec_level1 = in_dec_level1 + out_enc_level1
        out = self.decoder_level1(out)

        out = self.project_out(out.mean(0))

        out = self.left_scale_up_1_1(self.left_scale_up_0_1(out))

        out = out + shortcut

        return out