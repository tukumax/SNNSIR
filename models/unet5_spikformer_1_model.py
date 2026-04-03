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


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        # self.fc1_bn = nn.BatchNorm2d(hidden_features)
        # self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc1_body = nn.Sequential(
            Conv1x1(in_features, hidden_features),
            BN(hidden_features,v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        # self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        # self.fc2_bn = nn.BatchNorm2d(out_features)
        # self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.fc2_body = nn.Sequential(
            Conv1x1(hidden_features, out_features),
            BN(out_features,v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        # self.c_hidden = hidden_features
        # self.c_output = out_features

    def forward(self, x):
        # T,B,C,H,W = x.shape
        # x = self.fc1_conv(x)
        # x = self.fc1_bn(x)
        # x = self.fc1_lif(x)

        # x = self.fc2_conv(x)
        # x = self.fc2_bn(x)
        # x = self.fc2_lif(x)
        x = self.fc2_body(self.fc1_body(x))
        return x

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        # self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        # self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_body = nn.Sequential(
            Conv1dT(dim, dim),
            BN1d(dim, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        # self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        # self.k_bn = nn.BatchNorm1d(dim)
        # self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_body = nn.Sequential(
            Conv1dT(dim, dim),
            BN1d(dim,v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        # self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        # self.v_bn = nn.BatchNorm1d(dim)
        # self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.v_body = nn.Sequential(
            Conv1dT(dim, dim),
            BN1d(dim,v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        # self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.attn_lif = activation(v_threshold=v_th/2, decay_input=decay_input, v_reset=v_reset)

        # self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        # self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.proj_conv_body = nn.Sequential(
            Conv1dT(dim, dim),
            BN1d(dim,v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )
        # self.proj_conv = Linear(dim, dim)
        # self.proj_bn = BN1d(dim)
        # self.proj_lif = activation(v_threshold=v_th)
        self.matmul1 = SpikingMatmul('both')
        self.matmul2 = SpikingMatmul('l')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        # q_conv_out = self.q_conv(x)
        # q_conv_out = self.q_bn(q_conv_out)
        # q = self.q_lif(q_conv_out)
        q = self.q_body(x)
        q = q.transpose(-1,-2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()  # t b h n c//h

        # k_conv_out = self.k_conv(x)
        # k_conv_out = self.k_bn(k_conv_out)
        # k_conv_out = self.k_lif(k_conv_out)
        k = self.k_body(x)
        k = k.transpose(-1,-2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # v_conv_out = self.v_conv(x)
        # v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        # v_conv_out = self.v_lif(v_conv_out)
        v = self.v_body(x)
        v = v.transpose(-1,-2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = self.matmul1(k.transpose(-2, -1), v)
        x = self.matmul2(q, x) * self.scale  # t b h n c//h

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_conv_body(x).reshape(T, B, C, H, W)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, activation=LIF,v_th=0.2,alpha=(1/(2 ** 0.5)), decay_input=True, v_reset=0.0):
        super().__init__()
        ####
        self.head_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        ###
        self.attn = SSA(dim, num_heads=num_heads, activation=activation, v_th=v_th, alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, activation=activation, v_th=v_th, alpha=alpha, decay_input=decay_input, v_reset=v_reset)

    def forward(self, x):
        ###
        x = self.head_lif(x)
        ###
        x_attn = self.attn(x)
        x = x + x_attn
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
        self.skip = ['patch_embed', 'project_out']
        self.patch_embed = OverlapPatchEmbed(in_channels, planes[0])

        self.encoder_level1 = Block(planes[0], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.down1_2 = DownsampleLayer(planes[0], planes[1], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level2 = Block(planes[1], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.down2_3 = DownsampleLayer(planes[1], planes[2], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level3 = Block(planes[2], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.down3_4 = DownsampleLayer(planes[2], planes[3], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.encoder_level4 = Block(planes[3], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.down4_5 = DownsampleLayer(planes[3], planes[4], 2, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.scam = Block(planes[4], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.up5_4 = UpsampleLayer(planes[4], planes[3], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level4 = Block(planes[3], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.up4_3 = UpsampleLayer(planes[3], planes[2], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level3 = Block(planes[2], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.up3_2 = UpsampleLayer(planes[2], planes[1], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level2 = Block(planes[1], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.up2_1 = UpsampleLayer(planes[1], planes[0], 1, activation=activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)
        self.decoder_level1 = Block(planes[0], num_heads, mlp_ratio, activation,v_th=v_th,alpha=alpha, decay_input=decay_input, v_reset=v_reset)

        self.project_out = nn.Conv2d(planes[0], out_channels, 3, 1, 1, bias=False)


    def forward(self, x):
        shortcut = x.clone()

        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

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

        out = self.project_out(out_dec_level1.mean(0))
        out = out + shortcut

        return out