import argparse
import os
import time

import numpy as np
from thop import profile

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import DataParallel
def gblock(filnum_input,filnum, kernel=4):
    block = nn.Sequential(nn.Conv1d(in_channels=filnum_input, out_channels=filnum, kernel_size=kernel,stride=2,padding=1),nn.BatchNorm1d(filnum),nn.LeakyReLU(0.2))
    return block

def upblock(filnum_input,filnum, kernel=4):
    block = nn.Sequential(nn.ConvTranspose1d(in_channels=filnum_input, out_channels=filnum, kernel_size=kernel,stride=2,padding=1), nn.BatchNorm1d(filnum),nn.Dropout(0.1),nn.ReLU(0.2))
    return block

def upblock_second(filnum_input,filnum, kernel=4):
    block = nn.Sequential(nn.ConvTranspose1d(in_channels=filnum_input, out_channels=filnum, kernel_size=kernel,stride=2,padding=1), nn.BatchNorm1d(filnum),nn.ReLU(0.2))
    return block

def conv(input,output,kernel):
    block = nn.Sequential(nn.Conv1d(in_channels=input, out_channels=output, kernel_size=kernel,stride=2,padding=1),nn.LeakyReLU(0.2))
    return block

def upconv(input,output,kernel):
    block = nn.Sequential(nn.ConvTranspose1d(in_channels=input, out_channels=output, kernel_size=kernel,stride=2,padding=1),nn.LeakyReLU(0.2))
    return block

class Generator_gan(nn.Module):
    def __init__(self, length=1024,in_channels=1,out_channels=11):
        super(Generator_gan, self).__init__()
        self.gblock1 = gblock(64,128)
        self.gblock2 = gblock(128,256)
        self.gblock3 = gblock(256,512)
        self.gblock4 = gblock(512,512)
        self.gblock5 = gblock(512,512)
        self.gblock6 = gblock(512,512)
        self.gblock7 = gblock(512,1024)
        self.upblock1 = upblock(1024,512)
        self.upblock2 = upblock(1024,512)
        self.upblock3 = upblock(1024,512)
        self.upblock4 = upblock_second(1024,512)
        self.upblock5 = upblock_second(1024,256)
        self.upblock6 = upblock_second(512,128)
        self.upblock7 = upblock_second(256,64)
        self.conv = conv(in_channels,64,4)
        self.upconv = upconv(128,out_channels,4)


    def forward(self,z):
        x1 = self.conv(z)
        x2 = self.gblock1(x1)
        x3 = self.gblock2(x2)
        x4 = self.gblock3(x3)
        x5 = self.gblock4(x4)
        x6 = self.gblock5(x5)
        x7 = self.gblock6(x6)
        x8 = self.gblock7(x7)

        y1 = self.upblock1(x8)
        y1 = torch.cat((y1,x7),dim=1)
        y2 = self.upblock2(y1)
        y2 = torch.cat((y2,x6),dim=1)
        y3 = self.upblock3(y2)
        y3 = torch.cat((y3, x5), dim=1)
        y4 = self.upblock4(y3)
        y4 = torch.cat((y4, x4), dim=1)
        y5 = self.upblock5(y4)
        y5 = torch.cat((y5, x3), dim=1)
        y6 = self.upblock6(y5)
        y6 = torch.cat((y6, x2), dim=1)
        y7 = self.upblock7(y6)
        y7 = torch.cat((y7, x1), dim=1)

        output = self.upconv(y7)

        return output
class Discriminator_gan(nn.Module):
    def __init__(self, in_channels=11, length=1024):
        super(Discriminator_gan, self).__init__()
        self.down1 = gblock(11,256)
        self.down2 = gblock(256,128)
        self.down3 = gblock(128,64)
        self.down4 = gblock(64,32)
        self.down5 = gblock(32,32)
        self.down6 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=1, kernel_size=4,stride=2,padding=1),nn.Sigmoid())

    def forward(self,z):
        x1 = self.down1(z)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        return x6


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='DCGAN')
#     parser.add_argument('--train_epoch', type=int, default=200, help='epoch number of testing')
#     parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
#     parser.add_argument("--lr1", type=float, default=1e-4, help="adam: learning rate")
#     parser.add_argument("--lr2", type=float, default=1e-4, help="adam: learning rate")
#     parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
#     args = parser.parse_args()
#     print(args)
#     g_loss_list,d_loss_list,d_real_list,d_fake_list,mse,epoch_gloss_list,sim = [],[],[],[],[],[],[]
#     keys = ['g_loss_list', 'd_loss_list', 'd_real_list', 'd_fake_list', 'mse', 'epoch_gloss_list', 'sim']
#     record_dict = {
#         'g_loss_list': g_loss_list,
#         'd_loss_list': d_loss_list,
#         'd_real_list': d_real_list,
#         'd_fake_list': d_fake_list,
#         'mse': mse,
#         'epoch_gloss_list': epoch_gloss_list,
#         'sim': sim
#     }
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     generator = Generator_gan(in_channels=1,out_channels=11)
#     discriminator = Discriminator_gan(in_channels=11)
#     # Initialize weights
#     generator = generator.to(device)
#     discriminator = discriminator.to(device)
#     start_time = time.time()
#     for i in range(500):
#         generator(torch.ones((128, 1, 300)).to(device))
#     end_time = time.time()  # 记录结束时间
#     inference_time = (end_time - start_time) / 500 * 1000  # 计算推理时间
#     print(f'Inference Time: {inference_time} ms')
#
#     # with torch.no_grad():
#     #     flops, params = profile(generator, inputs=(torch.ones((38, 1, 1024)).to(device), ))
#     #     print("Total parameters:", params)
#     #     print("Total FLOPs:", flops)


