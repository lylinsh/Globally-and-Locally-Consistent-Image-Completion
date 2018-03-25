#coding=utf-8
import torch
import torch.nn as nn

class CompletionNet(nn.Module):
    def __init__(self):
        super(CompletionNet, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=4, dilation=4, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=8, dilation=8, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=16, dilation=16, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Tanh(),
                # nn.ReLU(True)
            )
    def forward(self, input_data):
        # print(input_data)
        return self.main(input_data)


class Discriminator(nn.Module):
    def __init__(self, global_size=256, local_size=128):
    # def __init__(self, x_ld, x_gd, global_size=256, local_size=128):
        l_size = local_size
        g_size = global_size

        for i in range(5):
            l_size = (l_size - 1)//2 + 1
        for i in range(6):
            g_size = (g_size - 1)//2 + 1

        super(Discriminator, self).__init__()
        layer_l1 = nn.Sequential()
        layer_l1.add_module('ld_c0', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True))
        layer_l1.add_module('ld_n0', nn.BatchNorm2d(64))
        layer_l1.add_module('ld_r0', nn.LeakyReLU(True))
        self.layer_l1 = layer_l1

        layer_l2 = nn.Sequential()
        layer_l2.add_module('ld_c1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=True))
        layer_l2.add_module('ld_n1', nn.BatchNorm2d(128))
        layer_l2.add_module('ld_r1', nn.LeakyReLU(True))
        self.layer_l2 = layer_l2

        layer_l3 = nn.Sequential()
        layer_l3.add_module('ld_c2', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=True))
        layer_l3.add_module('ld_n2', nn.BatchNorm2d(256))
        layer_l3.add_module('ld_r2', nn.LeakyReLU(True))
        self.layer_l3 = layer_l3

        layer_l4 = nn.Sequential()
        layer_l4.add_module('ld_c3', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True))
        layer_l4.add_module('ld_n3', nn.BatchNorm2d(512))
        layer_l4.add_module('ld_r3', nn.LeakyReLU(True))
        self.layer_l4 = layer_l4

        layer_l5 = nn.Sequential()
        layer_l5.add_module('ld_c4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True))
        layer_l5.add_module('ld_n4', nn.BatchNorm2d(512))
        layer_l5.add_module('ld_r4', nn.LeakyReLU(True))
        self.layer_l5 = layer_l5

        layer_lf = nn.Sequential()
        layer_lf.add_module('ld_f', nn.Linear(512*l_size*l_size, 1024))
        layer_lf.add_module('ld_rf', nn.LeakyReLU(True))
        self.layer_lf = layer_lf

        layer_g1 = nn.Sequential()
        layer_g1.add_module('gd_c0', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True))
        layer_g1.add_module('gd_n0', nn.BatchNorm2d(64))
        layer_g1.add_module('gd_r0', nn.LeakyReLU(True))
        self.layer_g1 = layer_g1

        layer_g2 = nn.Sequential()
        layer_g2.add_module('gd_c1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, bias=True))
        layer_g2.add_module('gd_n1', nn.BatchNorm2d(128))
        layer_g2.add_module('gd_r1', nn.LeakyReLU(True))
        self.layer_g2 = layer_g2

        layer_g3 = nn.Sequential()
        layer_g3.add_module('gd_c2', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=True))
        layer_g3.add_module('gd_n2', nn.BatchNorm2d(256))
        layer_g3.add_module('gd_r2', nn.LeakyReLU(True))
        self.layer_g3 = layer_g3

        layer_g4 = nn.Sequential()
        layer_g4.add_module('gd_c3', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True))
        layer_g4.add_module('gd_n3', nn.BatchNorm2d(512))
        layer_g4.add_module('gd_r3', nn.LeakyReLU(True))
        self.layer_g4 = layer_g4

        layer_g5 = nn.Sequential()
        layer_g5.add_module('gd_c4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True))
        layer_g5.add_module('gd_n4', nn.BatchNorm2d(512))
        layer_g5.add_module('gd_r4', nn.LeakyReLU(True))
        self.layer_g5 = layer_g5

        layer_g6 = nn.Sequential()
        layer_g6.add_module('gd_c4', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2, bias=True))
        layer_g6.add_module('gd_n4', nn.BatchNorm2d(512))
        layer_g6.add_module('gd_r4', nn.LeakyReLU(True))
        self.layer_g6 = layer_g6

        layer_gf = nn.Sequential()
        layer_gf.add_module('gd_f', nn.Linear(512*g_size*g_size, 1024))
        layer_gf.add_module('gd_rf', nn.LeakyReLU(True))
        self.layer_gf = layer_gf

        layer_ol = nn.Linear(2048, 1)
        self.layer_ol = layer_ol

        layer_sg = nn.Sigmoid()
        self.layer_sg = layer_sg

    def forward(self, x_ld, x_gd):
        ld = x_ld
        ld = self.layer_l1(ld)
        ld = self.layer_l2(ld)
        ld = self.layer_l3(ld)
        ld = self.layer_l4(ld)
        ld = self.layer_l5(ld)
        ld = ld.view(ld.size(0), -1)
        ld = self.layer_lf(ld)

        gd = x_gd
        gd = self.layer_g1(gd)
        gd = self.layer_g2(gd)
        gd = self.layer_g3(gd)
        gd = self.layer_g4(gd)
        gd = self.layer_g5(gd)
        gd = self.layer_g6(gd)
        gd = gd.view(gd.size(0), -1)
        gd = self.layer_gf(gd)

        com_gl = [ld, gd]
        layer_o = torch.cat(com_gl, 1)
        layer_o = self.layer_ol(layer_o)
        layer_o = self.layer_sg(layer_o)

        return layer_o
