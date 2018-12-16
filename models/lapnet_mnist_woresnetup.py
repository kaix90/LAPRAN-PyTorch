# y is not incorporated as the input for each hierarchy

import torch
import torch.nn as nn
import torch.nn.functional as F

# The implementation of gen1, input is 1x8x8
class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level1, self).__init__()

        self.input_channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 256, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, channels, kernel_size=1, stride=1)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# The implementation of the gen2, input is 3x8x8
class LAPGAN_Generator_level2(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level2, self).__init__()

        self.base = 128
        self.input_channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, self.base, kernel_size=1, stride=1),  # 64x8x8
            nn.BatchNorm2d(self.base),
            nn.ReLU(True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2),    # 128x4x4
            nn.BatchNorm2d(2*self.base),
            nn.ReLU(True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2),  # 256x2x2
            nn.BatchNorm2d(4*self.base),
            nn.ReLU(True),

            nn.ConvTranspose2d(4*self.base, 2*self.base, kernel_size=2, stride=2),  # 128x4x4
            nn.BatchNorm2d(2*self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*self.base, self.base, kernel_size=2, stride=2),   # 64x8x8
            nn.BatchNorm2d(self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base, self.input_channels, kernel_size=2, stride=2),     # 3x16x16
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# The implementation of the disc2, input is 1x16x16
class LAPGAN_Discriminator_level2(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level2, self).__init__()

        self.base = 64
        self.channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=2, stride=2, bias=False),  # 32x8x8
            nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2,bias=False),  # 64x4x4
            nn.BatchNorm2d(2*self.base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2, bias=False),  # 128x2x2
            nn.BatchNorm2d(4*self.base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*self.base, 1, kernel_size=2, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

# The implementation of the gen3, input is 1x16x16
class LAPGAN_Generator_level3(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level3, self).__init__()

        self.base = 256
        self.channels = channels
        self.ngpu = ngpu
        '''
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.channels, kernel_size=2, stride=2)
        )
        '''
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=1, stride=1),  # 64x16x16
            nn.BatchNorm2d(self.base),
            nn.ReLU(True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2), # 128x8x8
            nn.BatchNorm2d(2*self.base),
            nn.ReLU(True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2),    # 256x4x4
            nn.BatchNorm2d(4*self.base),
            nn.ReLU(True),

            nn.ConvTranspose2d(4*self.base, 2*self.base, kernel_size=2, stride=2),  # 128x8x8
            nn.BatchNorm2d(2*self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*self.base, self.base, kernel_size=2, stride=2),  # 64x16x16
            nn.BatchNorm2d(self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base, self.channels, kernel_size=2, stride=2), # 1x32x32
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# The implementation of the disc3, input is 1x32x32
class LAPGAN_Discriminator_level3(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level3, self).__init__()

        self.base = 16
        self.channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=2, stride=2, bias=False),  # 32x16x16
            nn.BatchNorm2d(self.base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2, bias=False),  # 64x8x8
            nn.BatchNorm2d(2*self.base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2, bias=False),  # 128x4x4
            nn.BatchNorm2d(4*self.base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4*self.base, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

# The implementation of the gen4, input is 1x32x32
class LAPGAN_Generator_level4(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Generator_level4, self).__init__()

        self.base = 64
        self.channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=3, stride=1, padding=1),  # 64x32x32
            nn.BatchNorm2d(self.base),
            nn.ReLU(True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=3, stride=2, padding=1), # 128x16x16
            nn.BatchNorm2d(2*self.base),
            nn.ReLU(True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=3, stride=2, padding=1),    # 256x8x8
            nn.BatchNorm2d(4*self.base),
            nn.ReLU(True),
            nn.Conv2d(4*self.base, 8*self.base, kernel_size=3, stride=2, padding=1),  # 512x4x4
            nn.BatchNorm2d(8*self.base),
            nn.ReLU(True),

            nn.ConvTranspose2d(8*self.base, 4*self.base, kernel_size=2, stride=2),  # 256x8x8
            nn.BatchNorm2d(4*self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*self.base, 2*self.base, kernel_size=2, stride=2),  # 128x16x16
            nn.BatchNorm2d(2*self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*self.base, self.base, kernel_size=2, stride=2),  # 64x32x32
            nn.BatchNorm2d(self.base),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base, self.channels, kernel_size=2, stride=2), # 1x64x64
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# The implementation of the disc4, input is 1x64x64
class LAPGAN_Discriminator_level4(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level4, self).__init__()

        self.base = 16
        self.channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=3, stride=2, padding=1, bias=False),  # 32x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=3, stride=2, padding=1, bias=False),  # 64x16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=3, stride=2, padding=1, bias=False),  # 128x8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*self.base, 8*self.base, kernel_size=3, stride=2, padding=1, bias=False),  # 256x4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*self.base, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

class LAPGAN(nn.Module):
    def __init__(self, channels, ngpu, LAPGAN_Generator_level1, LAPGAN_Generator_level2, LAPGAN_Generator_level3,
                 LAPGAN_Generator_level4):
        super(LAPGAN, self).__init__()

        self.channels = channels
        self.ngpu = ngpu
        self.LAPGAN_Generator_level1 = LAPGAN_Generator_level1
        self.LAPGAN_Generator_level2 = LAPGAN_Generator_level2
        self.LAPGAN_Generator_level3 = LAPGAN_Generator_level3
        self.LAPGAN_Generator_level4 = LAPGAN_Generator_level4

        self.main = nn.Sequential(self.LAPGAN_Generator_level1,
                                  self.LAPGAN_Generator_level2,
                                  self.LAPGAN_Generator_level3,
                                  self.LAPGAN_Generator_level4)

    def forward(self, input):
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output
