import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul, LIFSigmoid
from spikingjelly.activation_based import layer

alpha = 1 / (2 ** 0.5)


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, out_channels=48, bias=False, padding_mode='reflect'):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = Conv3x3(in_channels, out_channels, stride=1, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.proj(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect') -> None:
        super().__init__()
        self.body = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(in_channels, out_channels, stride=stride, padding_mode=padding_mode),
            BN(num_features=out_channels, v_th=v_th, alpha=alpha)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = 2
        self.up = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(in_channels, out_channels, stride=stride, padding_mode=padding_mode),
            BN(num_features=out_channels, v_th=v_th, alpha=alpha)
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


# spike residual basic block
class SRBB(nn.Module):
    def __init__(self, channels, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(SRBB, self).__init__()
        self.body = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, v_th=v_th, alpha=alpha)
        )

    def forward(self, x):
        out = self.body(x)
        x = out + x
        return x

## SEW-Residual basic block
class SEWRBB(nn.Module):
    def __init__(self, channels, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(SEWRBB, self).__init__()
        self.body = nn.Sequential(
            Conv3x3(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv3x3(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, v_th=v_th, alpha=alpha),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )
    def forward(self, x):
        out = self.body(x)
        x = out + x
        return x

# feature extract blcok
class FEB(nn.Module):
    def __init__(self, channels, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(FEB, self).__init__()
        self.body = nn.Sequential(
            SRBB(channels, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode),
            SRBB(channels, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        )

    def forward(self, x_l, x_r):
        x_l = self.body(x_l)
        x_r = self.body(x_r)
        return x_l, x_r


# spike separable convolution
class SSC(nn.Module):
    def __init__(
            self,
            in_channels,
            expansion_ratio=4,
            activation=LIF,
            decay_input=True,
            v_reset=0.,
            v_th=0.2, padding_mode='reflect'
    ):
        super().__init__()
        inner_channels = int(in_channels * expansion_ratio)
        self.lif1 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.pwconv1 = Conv1x1(in_channels, inner_channels, padding_mode=padding_mode)
        self.bn1 = BN(num_features=inner_channels, alpha=alpha, v_th=v_th)
        self.lif2 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.dwconv = Conv3x3(inner_channels, inner_channels, groups=inner_channels, padding_mode=padding_mode)
        self.bn2 = BN(num_features=inner_channels, alpha=alpha, v_th=v_th)
        self.lif3 = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.pwconv2 = Conv1x1(inner_channels, in_channels, padding_mode=padding_mode)
        self.bn3 = BN(num_features=in_channels, alpha=alpha, v_th=v_th)

    def forward(self, x):
        short_cut = x
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x))
        x = self.lif2(x)
        x = self.bn2(self.dwconv(x))
        x = self.lif3(x)
        x = self.bn3(self.pwconv2(x))
        return x + short_cut


### Adaptive Weight
class AdaptiveWeight(nn.Module):
    def __init__(self,
                 in_channels,
                 activation=LIF,
                 decay_input=True,
                 v_reset=0.,
                 v_th=0.2,
                 padding_mode='reflect'):
        super().__init__()
        self.body = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(in_channels * 2, in_channels, padding_mode=padding_mode),
            BN(num_features=in_channels, alpha=alpha, v_th=v_th)
        )

    def forward(self, x_l, x_r):
        short_cut_l, short_cut_r = x_l, x_r
        x = torch.cat((x_l, x_r), dim=2)
        x = self.body(x)
        x_l = x_l * x
        x_r = x_r * x
        return x_l + short_cut_l, x_r + short_cut_r


# spike stereo refinement block
class SSRB(nn.Module):
    def __init__(self, in_channels, expansion_ratio=4, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(SSRB, self).__init__()
        self.ssc = SSC(in_channels, expansion_ratio, activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        self.adpWht = AdaptiveWeight(in_channels, activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

    def forward(self, x_l_r):
        x_l, x_r = x_l_r[0], x_l_r[1]
        x_l = self.ssc(x_l)
        x_r = self.ssc(x_r)
        x_l, x_r = self.adpWht(x_l, x_r)
        return x_l, x_r


# spike channel Gating
class SCM(nn.Module):
    def __init__(self, in_channels, ratio=8, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(SCM, self).__init__()
        self.shared_mlp = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(in_channels, in_channels // ratio, padding_mode=padding_mode),
            BN(num_features=in_channels // ratio, alpha=alpha, v_th=v_th),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(in_channels // ratio, in_channels, padding_mode=padding_mode),
            BN(num_features=in_channels, alpha=alpha, v_th=v_th)
        )

    def forward(self, x):
        # y = self.avg_pool(x)
        t, b, c, h, w = x.shape
        avg_x = torch.mean(x, dim=(3, 4), keepdim=True)
        max_x = torch.max(torch.flatten(x, start_dim=3), dim=3, keepdim=True)[0].reshape(t, b, c, 1, 1)
        avg_y = self.shared_mlp(avg_x)
        max_y = self.shared_mlp(max_x)
        y = avg_y + max_y
        return x * y

# spike spatial Gating
class SSM(nn.Module):
    def __init__(self, in_channles, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(SSM, self).__init__()
        self.proj = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(2, 1, padding_mode=padding_mode),
            BN(num_features=1, alpha=alpha, v_th=v_th),
        )

    def forward(self, x):
        avg_x = torch.mean(x, dim=2, keepdim=True)
        max_x, _ = torch.max(x, dim=2, keepdim=True)
        y = torch.cat([avg_x, max_x], dim=2)  # t b 2 h w
        y = self.proj(y)
        return x * y


# spike stereo modulation
class SSCM(nn.Module):
    def __init__(self, in_channels, ratio=8, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(SSCM, self).__init__()
        self.channel_down = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(in_channels * 2, in_channels, padding_mode=padding_mode),
            BN(num_features=in_channels, alpha=alpha, v_th=v_th)
        )
        self.joint_weight = nn.Sequential(
            SCM(in_channels, ratio, activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode),
            SSM(in_channels, activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        )

    def forward(self, x_l, x_r):
        x = torch.cat([x_l, x_r], dim=2)
        x = self.channel_down(x)  # t b c h w
        x = self.joint_weight(x)
        out_l = x_l * x
        out_r = x_r * x
        return x_l + out_l, x_r + out_r


# spike stereo cross-attention
class SSCA(nn.Module):
    def __init__(self,
                 channels,
                 num_heads=8,
                 scale=4,
                 qkv_bias=False,
                 activation=LIF,
                 decay_input=True,
                 v_reset=0.,
                 v_th=0.2,
                 padding_mode='reflect'):
        super(SSCA, self).__init__()
        self.num_heads = num_heads

        self.l_head_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        self.r_head_lif = activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)

        self.l_qk_scu = nn.Sequential(
            Conv1x1(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, alpha=alpha, v_th=v_th),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )
        self.r_qk_scu = nn.Sequential(
            # activation(v_threshold=v_th),
            Conv1x1(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, alpha=alpha, v_th=v_th),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )
        self.l_v_scu = nn.Sequential(
            # activation(v_threshold=v_th),
            Conv1x1(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, alpha=alpha, v_th=v_th),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )
        self.r_v_scu = nn.Sequential(
            # activation(v_threshold=v_th),
            Conv1x1(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, alpha=alpha, v_th=v_th),
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset)
        )

        self.matmul1 = SpikingMatmul('both')
        self.matmul2 = SpikingMatmul('l')
        self.matmul3 = SpikingMatmul('both')
        self.matmul4 = SpikingMatmul('l')

        # self.attn_lif = activation(v_threshold=v_th)
        self.l_project_out = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, alpha=alpha, v_th=v_th)
        )
        self.r_project_out = nn.Sequential(
            activation(v_threshold=v_th, decay_input=decay_input, v_reset=v_reset),
            Conv1x1(channels, channels, padding_mode=padding_mode),
            BN(num_features=channels, alpha=alpha, v_th=v_th)
        )

    def forward(self, x_l, x_r):
        t, b, c, h, w = x_l.shape
        x_l_short_cut, x_r_short_cut = x_l.clone(), x_r.clone()
        x_l = self.l_head_lif(x_l)
        x_r = self.r_head_lif(x_r)

        l_qk = self.l_qk_scu(x_l).permute(0, 1, 3, 4, 2).contiguous()  # t b h w c
        r_qk = self.r_qk_scu(x_r).permute(0, 1, 3, 4, 2).contiguous()  # t b h w c

        l_v = self.l_v_scu(x_l).permute(0, 1, 3, 4, 2).contiguous()  # t b h w c
        r_v = self.r_v_scu(x_r).permute(0, 1, 3, 4, 2).contiguous()  # t b h w c

        r_attn = self.matmul1(r_qk.transpose(-1, -2), r_v)  # t b h c c
        out_r2l = self.matmul2(l_qk, r_attn).permute(0,1,4,2,3)  # t b h w c -> t b c h w

        l_attn = self.matmul3(l_qk.transpose(-1, -2), l_v)
        out_l2r = self.matmul4(r_qk, l_attn).permute(0,1,4,2,3)


        return self.l_project_out(out_r2l) * x_l_short_cut + x_l_short_cut, self.r_project_out(out_l2r) * x_r_short_cut + x_r_short_cut


class Block(nn.Module):
    def __init__(self, in_channels, ratio=8, activation=LIF, decay_input=True, v_reset=0., v_th=0.2, padding_mode='reflect'):
        super(Block, self).__init__()
        self.feb = FEB(channels=in_channels, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        self.sscm = SSCM(in_channels=in_channels, ratio=ratio, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

    def forward(self, x_l, x_r):
        x_l, x_r = self.feb(x_l, x_r)
        x_l, x_r = self.sscm(x_l, x_r)
        return x_l, x_r


class SNNSIR(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=48,
                 T=4,
                 planes=[32, 64, 96, 128, 160],
                 # en_num_blocks=[2,2,4],
                 # de_num_blocks=[2,2,4],
                 # num_heads=[1,4,8],
                 num_heads=8,
                 scale=2,
                 activation='LIF',
                 ratio=8,
                 refine_dim=48,
                 refine_block_num=5,
                 decay_input=True,
                 v_reset=0.,
                 v_th=0.20,
                 padding_mode='reflect',
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
        self.skip = ['patch_embed', 'project_out', 'refine_patch_embed', 'refine_project_out']
        self.patch_embed = OverlapPatchEmbed(in_channels, dim, padding_mode=padding_mode)

        self.encoder_level1 = Block(planes[0], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)
        self.down1_2 = DownsampleLayer(planes[0], planes[1], 2, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

        self.encoder_level2 = Block(planes[1], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)
        self.down2_3 = DownsampleLayer(planes[1], planes[2], 2, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

        self.encoder_level3 = Block(planes[2], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)
        self.down3_4 = DownsampleLayer(planes[2], planes[3], 2, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

        self.encoder_level4 = Block(planes[3], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)
        self.down4_5 = DownsampleLayer(planes[3], planes[4], 2, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

        self.ssca = SSCA(channels=planes[4], num_heads=num_heads, scale=scale, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)

        self.up5_4 = UpsampleLayer(planes[4], planes[3], 1, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        self.decoder_level4 = Block(planes[3], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)

        self.up4_3 = UpsampleLayer(planes[3], planes[2], 1, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        self.decoder_level3 = Block(planes[2], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)

        self.up3_2 = UpsampleLayer(planes[2], planes[1], 1, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        self.decoder_level2 = Block(planes[1], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)

        self.up2_1 = UpsampleLayer(planes[1], planes[0], 1, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode)
        self.decoder_level1 = Block(planes[0], ratio, activation, decay_input, v_reset, v_th=v_th, padding_mode=padding_mode)

        self.project_out = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False, padding_mode=padding_mode)

        ######### Refinement Network #########
        self.refine_patch_embed = OverlapPatchEmbed(in_channels, refine_dim, padding_mode=padding_mode)
        self.refinement_network = nn.Sequential(*[SSRB(refine_dim, scale, activation=activation, decay_input=decay_input, v_reset=v_reset, v_th=v_th, padding_mode=padding_mode) for _ in range(refine_block_num)])
        self.refine_project_out = nn.Conv2d(refine_dim, out_channels, 3, 1, 1, bias=False, padding_mode=padding_mode)


    def forward(self, x_l, x_r):
        shortcut_l, shortcut_r = x_l.clone(), x_r.clone()

        if len(x_l.shape) < 5:
            x_l = (x_l.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
            x_r = (x_r.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        ###deep feature extract###
        in_enc_level1_l, in_enc_level1_r = self.patch_embed(x_l), self.patch_embed(x_r)

        out_enc_level1_l, out_enc_level1_r = self.encoder_level1(in_enc_level1_l, in_enc_level1_r)
        in_enc_level2_l, in_enc_level2_r = self.down1_2(out_enc_level1_l), self.down1_2(out_enc_level1_r)

        out_enc_level2_l, out_enc_level2_r = self.encoder_level2(in_enc_level2_l, in_enc_level2_r)
        in_enc_level3_l, in_enc_level3_r = self.down2_3(out_enc_level2_l), self.down2_3(out_enc_level2_r)

        out_enc_level3_l, out_enc_level3_r = self.encoder_level3(in_enc_level3_l, in_enc_level3_r)
        in_enc_level4_l, in_enc_level4_r = self.down3_4(out_enc_level3_l), self.down3_4(out_enc_level3_r)

        out_enc_level4_l, out_enc_level4_r = self.encoder_level4(in_enc_level4_l, in_enc_level4_r)
        in_enc_level5_l, in_enc_level5_r = self.down4_5(out_enc_level4_l), self.down4_5(out_enc_level4_r)

        out_dec_level5_l, out_dec_level5_r = self.ssca(in_enc_level5_l, in_enc_level5_r)

        in_dec_level4_l, in_dec_level4_r = self.up5_4(out_dec_level5_l), self.up5_4(out_dec_level5_r)
        in_dec_level4_l = in_dec_level4_l + out_enc_level4_l
        in_dec_level4_r = in_dec_level4_r + out_enc_level4_r
        out_dec_level4_l, out_dec_level4_r = self.decoder_level4(in_dec_level4_l, in_dec_level4_r)

        in_dec_level3_l, in_dec_level3_r = self.up4_3(out_dec_level4_l), self.up4_3(out_dec_level4_r)
        in_dec_level3_l = in_dec_level3_l + out_enc_level3_l
        in_dec_level3_r = in_dec_level3_r + out_enc_level3_r
        out_dec_level3_l, out_dec_level3_r = self.decoder_level3(in_dec_level3_l, in_dec_level3_r)

        in_dec_level2_l, in_dec_level2_r = self.up3_2(out_dec_level3_l), self.up3_2(out_dec_level3_r)
        in_dec_level2_l = in_dec_level2_l + out_enc_level2_l
        in_dec_level2_r = in_dec_level2_r + out_enc_level2_r
        out_dec_level2_l, out_dec_level2_r = self.decoder_level2(in_dec_level2_l, in_dec_level2_r)

        in_dec_level1_l, in_dec_level1_r = self.up2_1(out_dec_level2_l), self.up2_1(out_dec_level2_r)
        in_dec_level1_l = in_dec_level1_l + out_enc_level1_l
        in_dec_level1_r = in_dec_level1_r + out_enc_level1_r
        out_dec_level1_l, out_dec_level1_r = self.decoder_level1(in_dec_level1_l, in_dec_level1_r)

        out_l, out_r = self.project_out(out_dec_level1_l.mean(0)), self.project_out(out_dec_level1_r.mean(0))
        out_l = out_l + shortcut_l
        out_r = out_r + shortcut_r

        ######### Refinement Network #########
        x_l = (out_l.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x_r = (out_r.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x_l, x_r = self.refine_patch_embed(x_l), self.refine_patch_embed(x_r)
        x_l, x_r = self.refinement_network((x_l, x_r))
        x_l, x_r = self.refine_project_out(x_l.mean(0)), self.refine_project_out(x_r.mean(0))
        refine_out_l = out_l + x_l
        refine_out_r = out_r + x_r

        return out_l, out_r, refine_out_l, refine_out_r