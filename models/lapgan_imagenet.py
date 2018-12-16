import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# The implementation of the network in the input
class LAPGAN_Generator_level1(nn.Module):
    def __init__(self, channels):
        super(LAPGAN_Generator_level1, self).__init__()

        self.input_channels = channels
        self.conv1 = nn.Conv2d(self.input_channels, 256, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.bn1(x)))
        x = self.conv3(self.bn2(x))


# The implementation of the generator of LAPGAN
class LAPGAN_Generator_level2(nn.Module):
    def __init__(self, leny):
        super(LAPGAN_Generator_level2, self).__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.linear = nn.Linear(256*4*4+leny, 256*4*4)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)

    def forward(self, x, y):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(self.bn1(res)))
        res = self.bn2(res)
        res = res.view(-1, 256*4*4)
        res = self.linear(torch.cat((res, y), 1))
        res = res.view(-1, 256*4*4)
        res = F.relu(self.conv3(res))
        res = self.conv4(self.bn3(res))
        x = x + res

class LAPGAN_Generator_level3(nn.Module):
    def __init__(self, leny):
        super(LAPGAN_Generator_level3, self).__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512*4*4 + leny, 512*4*4)
        self.conv4 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)

    def forward(self, x, y):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(self.bn1(res)))
        res = F.relu(self.conv3(self.bn2(res)))
        res = self.bn3(res)
        res = res.view(-1, 512*4*4)
        res = self.linear(torch.cat((res, y), 1))
        res = res.view(-1, 512*4*4)
        res = F.relu(self.conv4(res))
        res = F.relu(self.conv5(self.bn4(res)))
        res = self.conv6(self.bn5(res))
        x = x + res

class LAPGAN_Generator_level4(nn.Module):
    def __init__(self, leny):
        super(LAPGAN_Generator_level4, self).__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(1024)
        self.linear = nn.Linear(1024*4*4+leny, 1024*4*4)
        self.conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)

    def forward(self, x, y):
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(self.bn1(res)))
        res = F.relu(self.conv3(self.bn2(res)))
        res = F.relu(self.conv3(self.bn3(res)))
        res = self.bn4(res)
        res = res.view(-1, 1024*4*4)
        res = self.linear(torch.cat((res, y), 1))
        res = res.view(-1, 1024*4*4)
        res = F.relu(self.conv4(res))
        res = F.relu(self.conv5(self.bn4(res)))
        res = F.relu(self.conv6(self.bn5(res)))
        res = self.conv7(self.bn6(res))
        x = x + res


# The implementation of the discriminator of LAPGAN
class LAPGAN_Discriminator_level1(nn.Module):
    def __init__(self):
        super(LAPGAN_Discriminator_level1, self).__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(256*4*4, 1024)
        self.linear2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(self.bn1(x))
        x = self.linear1(x)
        x = self.linear2(x)

class LAPGAN_Discriminator_level2(nn.Module):
    def __init__(self):
        super(LAPGAN_Discriminator_level2, self).__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(512*4*4, 2048)
        self.linear2 = nn.Linear(2048, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.bn1(x)))
        x = self.con3(self.bn2(x))
        x = self.linear1(x)
        x = self.linear2(x)

class LAPGAN_Discriminator_level3(nn.Module):
    def __init__(self):
        super(LAPGAN_Discriminator_level3, self).__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(1024*4*4, 4096)
        self.linear2 = nn.Linear(4096, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.bn1(x)))
        x = F.relu(self.conv3(self.bn2(x)))
        x = self.conv4(self.bn3(x))
        x = self.linear1(x)
        x = self.linear2(x)
