import torch
import torch.nn as nn
import config

device = config.device


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, has_act=True, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.l_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        # self.elu = nn.ELU(inplace=True)
        self.has_act = has_act

    def forward(self, x):
        x = self.conv(x)
        if self.has_act:
            # x = self.l_relu(x)
            x = self.relu(x)
            # x = self.elu(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_size, out_size, has_act=True):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        # self.l_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        # self.elu = nn.ELU(inplace=True)
        self.has_act = has_act

    def forward(self, x):
        x = self.fc(x)
        if self.has_act:
            # x = self.l_relu(x)
            x = self.relu(x)
            # x = self.elu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding, has_act=False, bias=bias)
        # shortcut
        # self.conv_x = ConvBlock(in_channels, out_channels, 1, 1, 0, has_act=False, bias=bias)

    def forward(self, x):
        res = x
        res = self.conv1(res)
        res = self.conv2(res)
        return x + res


class CVAE(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block_num = config.block_num
        self.feature_num = config.feature_num
        print(f"config.cluster_num: {config.cluster_num}")

        # encoder
        self.encode_conv_input = ConvBlock(1, self.feature_num, kernel_size, stride, padding, has_act=False)
        self.encode_res_dense_blocks = self._make_res_blocks(self.block_num, self.feature_num, self.feature_num,
                                                             kernel_size, stride, padding)
        self.encode_conv_output = ConvBlock(self.feature_num, 1, kernel_size, stride, padding, has_act=False)
        # self.encode_linear_output = nn.Linear(config.input_size + config.cluster_num, config.latent_size)
        self.encode_mu = nn.Sequential(
            FCBlock(config.input_size + config.cluster_num, config.intermediate_size),
            FCBlock(config.intermediate_size, config.intermediate_size),
            # FCBlock(config.intermediate_size, config.intermediate_size),
            # FCBlock(config.intermediate_size, config.intermediate_size),
            FCBlock(config.intermediate_size, config.latent_size, has_act=False)
        )
        self.encode_log_var = nn.Sequential(
            FCBlock(config.input_size + config.cluster_num, config.intermediate_size),
            FCBlock(config.intermediate_size, config.intermediate_size),
            # FCBlock(config.intermediate_size, config.intermediate_size),
            # FCBlock(config.intermediate_size, config.intermediate_size),
            FCBlock(config.intermediate_size, config.latent_size, has_act=False)
        )

        # decoder
        self.decode_linear = nn.Sequential(
            FCBlock(config.latent_size + config.cluster_num, config.intermediate_size),
            FCBlock(config.intermediate_size, config.intermediate_size),
            # FCBlock(config.intermediate_size, config.intermediate_size),
            # FCBlock(config.intermediate_size, config.intermediate_size),
            FCBlock(config.intermediate_size, config.input_size, has_act=False)
        )

        self.decode_conv_input = ConvBlock(1, self.feature_num, kernel_size, stride, padding, has_act=False)
        self.decode_res_dense_blocks = self._make_res_blocks(self.block_num, self.feature_num, self.feature_num,
                                                             kernel_size, stride, padding)
        self.decode_conv_output = ConvBlock(self.feature_num, 1, kernel_size, stride, padding, has_act=False)

        # shortcut
        # self.conv_x = ConvBlock(in_channels, out_channels, 1, 1, 0, has_act=False, bias=bias)

    @staticmethod
    def _make_res_blocks(block_num, in_channels, out_channels, kernel_size, stride, padding):
        blocks = []
        for i in range(block_num):
            blocks.append(ResBlock(in_channels, out_channels, kernel_size, stride, padding))
        return nn.Sequential(*blocks)

    def encode(self, x, c):
        h_x = self.encode_conv_input(x)
        h_x = self.encode_res_dense_blocks(h_x)
        h_x = self.encode_conv_output(h_x)
        h_x = h_x.squeeze(1)
        h_x_c = torch.cat((h_x, c), dim=1)
        z_mu = self.encode_mu(h_x_c)
        z_log_var = self.encode_log_var(h_x_c)
        return z_mu, z_log_var

    def decode(self, z, c):
        z_c = torch.cat((z, c), dim=1)
        h_z_c = self.decode_linear(z_c)
        h_z_c = h_z_c.unsqueeze(1)
        h_z_c = self.decode_conv_input(h_z_c)
        h_z_c = self.decode_res_dense_blocks(h_z_c)
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

