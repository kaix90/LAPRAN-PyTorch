# y is not incorporated as the input for each hierarchy

import torch
import torch.nn as nn

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
            nn.Conv2d(64, channels, kernel_size=1, stride=1),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

class LAPGAN_Generator_level2(nn.Module):
    def __init__(self, channels, ngpu, leny):
        super(LAPGAN_Generator_level2, self).__init__()

        self.base = 64
        self.channels = channels
        self.ngpu = ngpu
        self.leny = leny

        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(self.channels, self.base, kernel_size=1, stride=1)  # 64x16x16
        self.bn1 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2)    # 128x8x8
        self.bn2 = nn.BatchNorm2d(2*self.base)
        self.conv3 = nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2)  # 256x4x4
        self.bn3 = nn.BatchNorm2d(4 * self.base)

        self.linear = nn.Linear(4*self.base*4*4 + self.leny, 4*self.base*4*4)
        self.dropout = nn.Dropout(0.5, inplace=True)

        self.deconv4 = nn.ConvTranspose2d(4*self.base, 2*self.base, kernel_size=2, stride=2)  # 128x8x8
        self.bn4 = nn.BatchNorm2d(2*self.base)
        self.deconv5 = nn.ConvTranspose2d(2 * self.base, self.channels, kernel_size=2, stride=2)  # 3x16x16
        self.tanh = nn.Tanh()

    def forward(self, input, y):
        output_up = self.upsamp(input)
        output = self.relu(self.bn1(self.conv1(output_up)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))

        output = torch.cat((output.view(-1, 4*self.base*4*4), y.view(-1, self.channels*8*8)), 1)
        output = self.linear(output)
        #output = self.dropout(output)
        output = output.view(-1, 4*self.base, 4, 4)
        output = self.relu(self.bn4(self.deconv4(output)))
        output = self.deconv5(output)
        output = self.tanh(output)
        output = output + output_up

        return output

# The implementation of the disc2, input is 1x16x16
class LAPGAN_Discriminator_level2(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level2, self).__init__()

        self.base = 32
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
    def __init__(self, channels, ngpu, leny):
        super(LAPGAN_Generator_level3, self).__init__()

        self.base = 64
        self.channels = channels
        self.ngpu = ngpu

        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(self.channels, self.base, kernel_size=1, stride=1)  # 64x32x32
        self.bn1 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2) # 128x16x16
        self.bn2 = nn.BatchNorm2d(2*self.base)
        self.conv3 = nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2)    # 256x8x8
        self.bn3 = nn.BatchNorm2d(4*self.base)
        self.conv4 = nn.Conv2d(4*self.base, 8*self.base, kernel_size=2, stride=2)    # 512x4x4
        self.linear = nn.Linear(8*self.base*4*4+leny, 8*self.base*4*4)
        self.dropout = nn.Dropout(0.5, inplace=True)

        self.deconv5 = nn.ConvTranspose2d(8 * self.base, 4 * self.base, kernel_size=2, stride=2)  # 256x8x8
        self.bn5 = nn.BatchNorm2d(4 * self.base)
        self.deconv6 = nn.ConvTranspose2d(4 * self.base, 2 * self.base, kernel_size=2, stride=2)  # 128x16x16
        self.bn6 = nn.BatchNorm2d(2 * self.base)
        self.deconv7 = nn.ConvTranspose2d(2 * self.base, self.channels, kernel_size=2, stride=2)    # 3x32x32
        self.tanh = nn.Tanh()

    def forward(self, input, y):
        output_up = self.upsamp(input)
        output = self.relu(self.bn1(self.conv1(output_up)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))
        output = self.conv4(output)
        output = torch.cat((output.view(-1, 8 * self.base * 4 * 4), y.view(-1, self.channels * 8 * 8)), 1)
        output = self.linear(output)
        #output = self.dropout(output)
        output = output.view(-1, 8 * self.base, 4, 4)
        output = self.relu(self.bn5(self.deconv5(output)))
        output = self.relu(self.bn6(self.deconv6(output)))
        output = self.deconv7(output)
        output = self.tanh(output)
        output = output + output_up

        return output

# The implementation of the disc3, input is 1x32x32
class LAPGAN_Discriminator_level3(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level3, self).__init__()

        self.base = 32
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

# The implementation of the gen4, input is 3x32x32
class LAPGAN_Generator_level4(nn.Module):
    def __init__(self, channels, ngpu, leny):
        super(LAPGAN_Generator_level4, self).__init__()

        self.base = 64
        self.channels = channels
        self.ngpu = ngpu

        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(self.channels, self.base, kernel_size=1, stride=1)  # 64x64x64
        self.bn1 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.base, 2 * self.base, kernel_size=2, stride=2)  # 128x32x32
        self.bn2 = nn.BatchNorm2d(2 * self.base)
        self.conv3 = nn.Conv2d(2 * self.base, 4 * self.base, kernel_size=2, stride=2)  # 256x16x16
        self.bn3 = nn.BatchNorm2d(4 * self.base)
        self.conv4 = nn.Conv2d(4 * self.base, 8 * self.base, kernel_size=2, stride=2)  # 512x8x8
        self.bn4 = nn.BatchNorm2d(8 * self.base)
        self.conv5 = nn.Conv2d(8 * self.base, 16 * self.base, kernel_size=2, stride=2)  # 1024x4x4
        self.linear = nn.Linear(16 * self.base * 4 * 4 + leny, 16 * self.base * 4 * 4)
        self.dropout = nn.Dropout(0.5, inplace=True)

        self.deconv5 = nn.ConvTranspose2d(16 * self.base, 8 * self.base, kernel_size=2, stride=2)  # 512x8x8
        self.bn5 = nn.BatchNorm2d(8 * self.base)
        self.deconv6 = nn.ConvTranspose2d(8 * self.base, 4 * self.base, kernel_size=2, stride=2)  # 256x16x16
        self.bn6 = nn.BatchNorm2d(4 * self.base)
        self.deconv7 = nn.ConvTranspose2d(4 * self.base, 2 * self.base, kernel_size=2, stride=2)  # 128x32x32
        self.bn7 = nn.BatchNorm2d(2 * self.base)
        self.deconv8 = nn.ConvTranspose2d(2 * self.base, self.channels, kernel_size=2, stride=2)  # 3x64x64
        self.tanh = nn.Tanh()

    def forward(self, input, y):
        output_up =self.upsamp(input)
        output = self.relu(self.bn1(self.conv1(output_up)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))
        output = self.relu(self.bn4(self.conv4(output)))
        output = self.conv5(output)
        output = torch.cat((output.view(-1, 16 * self.base * 4 * 4), y.view(-1, self.channels * 8 * 8)), 1)
        output = self.linear(output)
        #output = self.dropout(output)
        output = output.view(-1, 16 * self.base, 4, 4)
        output = self.relu(self.bn5(self.deconv5(output)))
        output = self.relu(self.bn6(self.deconv6(output)))
        output = self.relu(self.bn7(self.deconv7(output)))
        output = self.deconv8(output)
        output = self.tanh(output)
        output = output + output_up

        return output

# The implementation of the disc4, input is 1x64x64
class LAPGAN_Discriminator_level4(nn.Module):
    def __init__(self, channels, ngpu):
        super(LAPGAN_Discriminator_level4, self).__init__()

        self.base = 32
        self.channels = channels
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(self.channels, self.base, kernel_size=2, stride=2, bias=False),  # 32x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base, 2*self.base, kernel_size=2, stride=2, bias=False),  # 64x16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*self.base, 4*self.base, kernel_size=2, stride=2, bias=False),  # 128x8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*self.base, 8*self.base, kernel_size=2, stride=2, bias=False),  # 256x4x4
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

    def forward(self, input, y):
        output = self.LAPGAN_Generator_level1(input)
        output = self.LAPGAN_Generator_level2(output, y)
        output = self.LAPGAN_Generator_level3(output, y)
        output = self.LAPGAN_Generator_level4(output, y)

        return output
