import time

import torch
from thop import profile
from torch import nn


class ecgTransForm(nn.Module):
    def __init__(self):
        super(ecgTransForm, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(1, 32, kernel_size=filter_sizes[0],
                               stride=1, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(1, 32, kernel_size=filter_sizes[1],
                               stride=1, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(1, 32, kernel_size=filter_sizes[2],
                               stride=1, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(0.2)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 32 * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32 * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32 * 2, 128, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=40, nhead=5,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(1*128, 5)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        x_in = x_in.unsqueeze(dim=1)
        # Multi-scale Convolutions
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)

        # Bi-directional Transformer
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(torch.flip(x, [2]))
        x = x1 + x2

        x = self.aap(x)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# model = ecgTransForm()
# ecgs = torch.randn((128, 1, 300))
#
# start_time = time.time()
# for i in range(100):
#     output = model(ecgs)
# end_time = time.time()  # 记录结束时间
# inference_time = (end_time - start_time) / 100 * 1000  # 计算推理时间
# print(f'Inference Time: {inference_time} ms')

# with torch.no_grad():
#     flops, params = profile(model, inputs=(ecgs, ))
#     print("Total parameters:", params)
#     print("Total FLOPs:", flops)