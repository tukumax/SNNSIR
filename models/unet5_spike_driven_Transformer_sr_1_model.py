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
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0) -> None:
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


class MS_MLP_Conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_body = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(in_features, hidden_features),
            BN(hidden_features, v_th=v_th, alpha=alpha)
        )

        self.fc2_body = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(hidden_features, out_features),
            BN(out_features, v_th=v_th, alpha=alpha)
        )


    def forward(self, x):

        x = self.fc2_body(self.fc1_body(x))
        return x


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        activation=LIF, v_th=0.2, alpha=(1 / (2 ** 0.5)),
        decay_input=True, v_reset=0.0
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.scale = 0.125
        self.q_body = nn.Sequential(
            Conv1x1(dim, dim),
            BN(dim, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        self.k_body = nn.Sequential(
            Conv1x1(dim, dim),
            BN(dim, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        self.v_body = nn.Sequential(
            Conv1x1(dim, dim),
            BN(dim, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )


        # self.attn_lif = activation(v_threshold=v_th/2)
        # self.talking_heads = Conv1x1(num_heads, num_heads)
        self.talking_heads_lif = activation(v_threshold=v_th/2, decay_input=decay_input, v_reset=v_reset)

        self.proj_conv_body = nn.Sequential(
            Conv1x1(dim, dim),
            BN(dim, v_th=v_th, alpha=alpha)
        )

        self.shortcut_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)


    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.shortcut_lif(x)

        q = self.q_body(x)
        q = (
            q.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_body(x)
        k = (
            k.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_body(x)
        v = (
            v.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)  # T B head N c//h(1), head=c=d

        kv = kv.sum(dim=-2, keepdim=True)  # T B head 1 c//h
        kv = self.talking_heads_lif(kv)

        x = q.mul(kv)  # element-wise product


        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = self.proj_conv_body(x)

        x = x + identity
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super().__init__()
        self.attn = MS_SSA_Conv(dim, num_heads=num_heads, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

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
                 activation='LIF',
                 mlp_ratio=4,
                 v_th=0.2,
                 alpha=1 / (2 ** 0.5),
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

        self.encoder_level1 = Block(planes[0], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        # self.down1_2 = DownsampleLayer(planes[0], planes[1], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level2 = Block(planes[1], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        # self.down2_3 = DownsampleLayer(planes[1], planes[2], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level3 = Block(planes[2], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        # self.down3_4 = DownsampleLayer(planes[2], planes[3], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level4 = Block(planes[3], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        # self.down4_5 = DownsampleLayer(planes[3], planes[4], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.scam = Block(planes[4], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        # self.up5_4 = UpsampleLayer(planes[4], planes[3], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level4 = Block(planes[3], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        # self.up4_3 = UpsampleLayer(planes[3], planes[2], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level3 = Block(planes[2], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        # self.up3_2 = UpsampleLayer(planes[2], planes[1], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level2 = Block(planes[1], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        # self.up2_1 = UpsampleLayer(planes[1], planes[0], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level1 = Block(planes[0], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.project_out = nn.Conv2d(planes[0], out_channels, 3, 1, 1, bias=False)

        #
        self.sr_scale = sr_scale
        self.left_scale_up_0_1 = nn.Conv2d(out_channels, out_channels * (sr_scale ** 2), kernel_size=3, stride=1,
                                           padding=1, bias=False)
        self.left_scale_up_1_1 = nn.PixelShuffle(sr_scale)


    def forward(self, x):
        # shortcut = x.clone()
        shortcut = F.interpolate(x, scale_factor=self.sr_scale, mode='bicubic',align_corners=False)

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