import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torch.autograd import Variable
import math

class ssim_loss(torch.nn.Module):
    def __init__(self, max_val=1.0, k1=0.01, k2=0.03):
        super(ssim_loss, self).__init__()
        self.max_val = max_val
        self.k1 = k1
        self.k2 = k2

        # define kernels
        self.w_x = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        self.w_xx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        self.w_xy = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)

        self.w_x.weight.data += abs(self.w_x.weight.min().data[0])

    def forward(self, img_x, img_y):
        # check dimension mismatch
        if img_x.data.shape != img_y.data.shape:
            raise RuntimeError(
                'input images must have the same dimension {0:%d}, {1:%d}'.format(img_x.shape, img_y.shape))

        self.conv1 = self.w_x(img_x)
        self.conv2 = self.w_x(img_y)
        self.conv3 = self.w_x(img_x * img_x)
        self.conv4 = self.w_x(img_y * img_y)
        self.conv5 = self.w_x(img_x * img_y)

        shape = self.conv1.data.shape

        '''
        self.mu_x = torch.mean(self.conv1.view(shape[0], shape[1], -1), dim=(2))
        self.mu_y = torch.mean(self.conv2.view(shape[0], shape[1], -1), dim=(2))
        self.sigma_x = torch.mean(self.conv3.view(shape[0], shape[1], -1), dim=(2)) - self.mu_x ** 2
        self.sigma_y = torch.mean(self.conv4.view(shape[0], shape[1], -1), dim=(2)) - self.mu_y ** 2
        self.sigma_xy = torch.mean(self.conv5.view(shape[0], shape[1], -1), dim=(2)) - self.mu_x * self.mu_y
        '''

        self.mu_x = self.conv1
        self.mu_y = self.conv2
        self.sigma_x = self.conv3 - self.mu_x ** 2
        self.sigma_y = self.conv4 - self.mu_y ** 2
        self.sigma_xy = self.conv5 - self.mu_x * self.mu_y


        self.C1 = (self.k1 * self.max_val) ** 2
        self.C2 = (self.k2 * self.max_val) ** 2

        self.l = (2 * self.mu_x * self.mu_y + self.C1) / (self.mu_x ** 2 + self.mu_y ** 2 + self.C1)
        self.c = (2 * self.sigma_xy + self.C2) / (self.sigma_x ** 2 + self.sigma_y ** 2 + self.C2)

        #self.l = (2 * self.mu_x * self.mu_y) / (self.mu_x ** 2 + self.mu_y ** 2)
        #self.c = (2 * self.sigma_xy) / (self.sigma_x + self.sigma_y)

        self.s = 1 - torch.mean(self.l * self.c)

        return self.s

'''
class msssim_loss(torch.nn.Module):
    def __init__(self, img_x, img_y):

    def forward(self, img_x, img_y):
'''

if __name__ == "__main__":
    img_path = '/home/user/kaixu/myGitHub/datasets/BSDS500/train-aug/1'
    img_x_path = os.path.join(img_path, '2092_rot0.bmp')
    img_y_path = os.path.join(img_path, '12074_rot0.bmp')
    img_z_path = os.path.join(img_path, '8049_rot0.bmp')

    img_x = cv2.imread(img_x_path, 1).astype('float') / 255.0
    img_y = cv2.imread(img_y_path, 1).astype('float') / 255.0
    img_z = cv2.imread(img_z_path, 1).astype('float') / 255.0

    img_x = np.expand_dims(img_x.transpose([2, 0, 1]), axis=0)
    img_y = np.expand_dims(img_y.transpose([2, 0, 1]), axis=0)
    img_z = np.expand_dims(img_z.transpose([2, 0, 1]), axis=0)

    img_x_tensor = torch.FloatTensor(img_x)
    img_y_tensor = torch.FloatTensor(img_y)
    img_z_tensor = torch.FloatTensor(img_z)

    img_x_var = Variable(img_x_tensor)
    img_y_var = Variable(img_y_tensor)
    img_z_var = Variable(img_z_tensor)

    ssim_loss = ssim_loss()

    ssim_xx = ssim_loss(img_x_var, img_x_var)
    ssim_xy = ssim_loss(img_x_var, img_y_var)
    ssim_xz = ssim_loss(img_x_var, img_z_var)

    print("ssim_xx:{0}, ssim_xy:{1}, ssim_xz:{2}".format(ssim_xx.data[0], ssim_xy.data[0], ssim_xz.data[0]))