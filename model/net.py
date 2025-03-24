import torch
import torch.nn as nn

class Partial_conv(nn.Module):
    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div  # 2
        self.n_div = n_div     # 4
        self.dim_untouched = dim - self.dim_conv3  # 6
        self.partial_conv = nn.Conv1d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.channel_conv = nn.Conv1d(self.dim_untouched, self.dim_untouched, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        conv = x2.shape[-1] // self.n_div
        unconv = x2.shape[-1] - conv
        x3, x4 = torch.split(x2, [conv, unconv], dim=-1)
        x3 = self.channel_conv(x3)
        x2 = torch.cat((x3, x4), -1)
        x = torch.cat((x1, x2), 1)

        return x
#
#
# class DepthwiseSeparableConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(DepthwiseSeparableConv1d, self).__init__()
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
#         self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


class Model(nn.Module):
    def __init__(self, width_multiplier=0.5, resolution_multiplier=0.5):
        nn.Module.__init__(self)
        num_channels = [1,
                        1,
                        round(8 * width_multiplier * resolution_multiplier),
                        round(16 * width_multiplier * resolution_multiplier)]

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = Partial_conv(8, 4)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.BN = nn.BatchNorm1d(1)
        self.flatten = nn.Flatten()  # 扁平化层
        fc_input_size = num_channels[3] * 38

        # Define the fully connected layers with reduced out_features
        self.fc1 = nn.Sequential(
            nn.Linear(150, 16),  # Reduced out_features to 16
            nn.ReLU()  # 激活函数
        )
        self.fc2 = nn.Linear(16, 5)  # Reduced out_features to 5

    def forward(self, x):
        x = x.view(-1, 1, 300)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.pool1(x)
        x = self.BN(x)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
