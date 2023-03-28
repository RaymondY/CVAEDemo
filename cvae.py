import torch
import torch.nn as nn
from config import DefaultConfig

config = DefaultConfig()
device = config.device


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, has_act=True, bias=True):
        super().__init__()
        padding = kernel_size >> 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        # self.elu = nn.ELU()
        self.has_act = has_act

    def forward(self, x):
        x = self.conv(x)
        if self.has_act:
            x = self.relu(x)
            # x = self.elu(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, has_act=True):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
        # self.elu = nn.ELU()
        self.has_act = has_act

    def forward(self, x):
        x = self.fc(x)
        if self.has_act:
            x = self.relu(x)
            # x = self.elu(x)
        return x


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
#         super().__init__()
#         self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#         self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding, has_act=False, bias=bias)
#         # shortcut
#         # self.conv_x = ConvBlock(in_channels, out_channels, 1, 1, 0, has_act=False, bias=bias)
#
#     def forward(self, x):
#         res = x
#         res = self.conv1(res)
#         res = self.conv2(res)
#         return x + res


class FlattenLayer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            FlattenLayer(),
            LinearBlock(gate_channels, gate_channels // reduction_ratio),
            LinearBlock(gate_channels // reduction_ratio, gate_channels, has_act=False)
        )

    def forward(self, x):
        avg_pool = nn.AvgPool1d(x.size(2), stride=x.size(2))(x)
        channel_att_average = self.mlp(avg_pool)
        max_pool = nn.MaxPool1d(x.size(2), stride=x.size(2))(x)
        channel_att_max = self.mlp(max_pool)

        channel_att_sum = channel_att_average + channel_att_max
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)

        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        # kernel_size = 7
        kernel_size = 3
        self.spatial = ConvBlock(2, 1, kernel_size, 1, has_act=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

    @staticmethod
    def compress(x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# Channel Attention Module (CAM) and Spartial Attention Module (SAM)
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, has_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.has_spatial = has_spatial
        if has_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.has_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, reduction_ratio=16, has_spatial=True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, has_act=False)
        self.cbam = CBAM(out_channels, reduction_ratio, has_spatial)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.cbam(res)
        # return nn.ReLU()(x + res)
        return x + res


class CVAE(nn.Module):
    def __init__(self, cluster_num, kernel_size=3, stride=1):
        super().__init__()
        self.cluster_num = cluster_num
        self.block_num = config.block_num
        self.feature_num = config.feature_num

        # encoder
        self.encode_conv_input = ConvBlock(1, self.feature_num, kernel_size, stride, has_act=False)
        self.encode_cbam_blocks = self._make_cbam_blocks(self.block_num, self.feature_num, self.feature_num,
                                                         kernel_size, stride)
        # self.encode_res_dense_blocks = self._make_res_blocks(self.block_num, self.feature_num, self.feature_num,
        #                                                      kernel_size, stride, padding)
        self.encode_conv_output = ConvBlock(self.feature_num, 1, kernel_size, stride, has_act=False)
        self.encode_mu = nn.Sequential(
            LinearBlock(config.input_size + self.cluster_num, config.intermediate_size),
            LinearBlock(config.intermediate_size, config.intermediate_size),
            LinearBlock(config.intermediate_size, config.latent_size, has_act=False)
        )
        self.encode_log_var = nn.Sequential(
            LinearBlock(config.input_size + self.cluster_num, config.intermediate_size),
            LinearBlock(config.intermediate_size, config.intermediate_size),
            LinearBlock(config.intermediate_size, config.latent_size, has_act=False)
        )

        # decoder
        self.decode_mu = nn.Sequential(
            LinearBlock(config.latent_size + self.cluster_num, config.intermediate_size),
            LinearBlock(config.intermediate_size, config.intermediate_size),
            LinearBlock(config.intermediate_size, config.input_size, has_act=False)
        )

        self.decode_conv_input = ConvBlock(1, self.feature_num, kernel_size, stride, has_act=False)
        # self.decode_res_dense_blocks = self._make_res_blocks(self.block_num, self.feature_num, self.feature_num,
        #                                                      kernel_size, stride, padding)
        self.decode_cbam_blocks = self._make_cbam_blocks(self.block_num, self.feature_num, self.feature_num,
                                                         kernel_size, stride)
        self.decode_conv_output = ConvBlock(self.feature_num, 1, kernel_size, stride, has_act=False)

        # shortcut
        # self.conv_x = ConvBlock(in_channels, out_channels, 1, 1, 0, has_act=False, bias=bias)

    # @staticmethod
    # def _make_res_blocks(block_num, in_channels, out_channels, kernel_size, stride, padding):
    #     blocks = []
    #     for i in range(block_num):
    #         blocks.append(ResBlock(in_channels, out_channels, kernel_size, stride, padding))
    #     return nn.Sequential(*blocks)

    @staticmethod
    def _make_cbam_blocks(block_num, in_channels, out_channels, kernel_size, stride):
        blocks = []
        for i in range(block_num):
            blocks.append(CBAMResBlock(in_channels, out_channels, kernel_size, stride))
        return nn.Sequential(*blocks)

    def encode(self, x, c):
        h_x = self.encode_conv_input(x)
        h_x = self.encode_cbam_blocks(h_x)
        h_x = self.encode_conv_output(h_x)
        # h_x = x + h_x
        h_x = h_x.squeeze(1)
        h_x_c = torch.cat((h_x, c), dim=1)
        z_mu = self.encode_mu(h_x_c)
        z_log_var = self.encode_log_var(h_x_c)
        return z_mu, z_log_var

    def decode(self, z, c):
        z_c = torch.cat((z, c), dim=1)
        h_z_c = self.decode_mu(z_c)
        h_z_c = h_z_c.unsqueeze(1)
        h_z_c = self.decode_conv_input(h_z_c)
        h_z_c = self.decode_cbam_blocks(h_z_c)
        pred_x = self.decode_conv_output(h_z_c)
        return pred_x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(device)
        return eps * std + mu

    def forward(self, x, c):
        z_mu, z_log_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_log_var)
        pred_x = self.decode(z, c)
        return pred_x, z_mu, z_log_var

    def generate(self, c):
        # sample z from N(0, 1)
        z = torch.randn((c.shape[0], config.latent_size)).to(device)
        pred_x = self.decode(z, c)
        return pred_x
