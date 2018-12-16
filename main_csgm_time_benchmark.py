from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from numpy.random import randn
from torch.autograd import Variable
from torch.nn import init
from torchvision import datasets, transforms
import skimage.io as sio
import time

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet',
                    default='csgm')
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='cifar10')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--image-size', type=int, default=64, metavar='N',
                    help='The height / width of the input image to the network')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--layers-gan', type=int, default=3, metavar='N',
                    help='number of hierarchies in the GAN (default: 64)')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.01, metavar='N.',
                    help='penalty for the mse and bce loss')
parser.add_argument('--cr', type=int, default=30, help='compression ratio')

opt = parser.parse_args()
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: please run with GPU")
print(opt)

torch.cuda.set_device(opt.gpu)
print('Current gpu device: gpu %d' % (torch.cuda.current_device()))

if opt.seed is None:
    opt.seed = np.random.randint(1, 10000)
print('Random seed: ', opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

if not os.path.exists('%s/%s/cr%s/%s/test' % (opt.outf, opt.dataset, opt.cr, opt.model)):
    os.makedirs('%s/%s/cr%s/%s/test' % (opt.outf, opt.dataset, opt.cr, opt.model))

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def data_loader():
    kwopt = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}
    if opt.dataset == 'bsd500_patch':
        test_dataset = datasets.ImageFolder(root=opt.datapath + 'val_64x64',
                                            transform=transforms.Compose([
                                                transforms.Resize(opt.image_size),
                                                transforms.CenterCrop(opt.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
    elif opt.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(opt.image_size),
                                            transforms.CenterCrop(opt.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    elif opt.dataset == 'mnist':
        test_dataset = datasets.MNIST('./data', train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.image_size),
                                         transforms.CenterCrop(opt.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, **kwopt)

    return test_loader

class netG(nn.Module):
    def __init__(self, channels, leny):
        super(netG, self).__init__()

        self.channels = channels
        self.base = 64
        self.fs = 4
        self.leny = leny

        self.linear1 = nn.Linear(self.channels * self.leny, self.base * 8 * self.fs ** 2)
        self.bn1 = nn.BatchNorm2d(self.base * 8)
        self.deconv2 = nn.ConvTranspose2d(self.base * 8, self.base * 4, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.base * 4)
        self.relu = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(self.base * 4, self.base * 2, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.base * 2)
        self.deconv4 = nn.ConvTranspose2d(self.base * 2, self.base, kernel_size=4, padding=1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(self.base)
        self.deconv5 = nn.ConvTranspose2d(self.base, self.channels, kernel_size=4, padding=1, stride=2, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.linear1(self.output)
        self.output = self.output.view(self.output.size(0), self.base * 8, self.fs, self.fs)
        self.output = self.relu(self.bn1(self.output))
        self.output = self.relu(self.bn2(self.deconv2(self.output)))
        self.output = self.relu(self.bn3(self.deconv3(self.output)))
        self.output = self.relu(self.bn4(self.deconv4(self.output)))
        self.output = self.deconv5(self.output)
        self.output = self.tanh(self.output)

        return self.output


class netD(nn.Module):
    def __init__(self, channels):
        super(netD, self).__init__()

        self.channels = channels
        self.base = 64

        self.conv1 = nn.Conv2d(self.channels, self.base, 4, 2, 1, bias=False)  # 64 x 32 x 32
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.base * 2, 4, 2, 1, bias=False)  # 128 x 16 x 16
        self.bn2 = nn.BatchNorm2d(self.base * 2)

        self.conv3 = nn.Conv2d(self.base * 2, self.base * 4, 4, 2, 1, bias=False)  # 256 x 8 x 8
        self.bn3 = nn.BatchNorm2d(self.base * 4)

        self.conv4 = nn.Conv2d(self.base * 4, self.base * 8, 4, 2, 1, bias=False)  # 512 x 4 x 4
        self.bn4 = nn.BatchNorm2d(self.base * 8)

        self.linear1 = nn.Linear(self.base * 8 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        self.output = self.lrelu(self.conv1(input))
        self.output = self.lrelu(self.bn2(self.conv2(self.output)))
        self.output = self.lrelu(self.bn3(self.conv3(self.output)))
        self.output = self.lrelu(self.bn4(self.conv4(self.output)))
        self.output = self.output.view(self.output.size(0), -1)
        self.output = self.sigmoid(self.linear1(self.output))

        return self.output

def evaluation(testloader):
    # Initialize variables
    input, _ = testloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    channels = sz_input[1]
    img_size = sz_input[2]
    n = img_size ** 2
    m = n / opt.cr

    sensing_matrix = randn(channels, m, n)

    input = torch.FloatTensor(opt.batch_size, channels, m)

    # Instantiate models
    gen = netG(channels, m)

    if opt.dataset == 'cifar10':
        if opt.cr == 5:
            level1_iter = 10 # 0.0093
        elif opt.cr == 10:
            level1_iter = 15 # 0.0093
        elif opt.cr == 20:
            level1_iter = 23 # 0.0120
        elif opt.cr == 30:
            level1_iter = 32 # 0.0142
    elif opt.dataset == 'mnist':
        if opt.cr == 5:
            level1_iter = 16 # 0.0055
        elif opt.cr == 10:
            level1_iter = 8 # 0.0077
        elif opt.cr == 20:
            level1_iter = 41 # 0.0077
        elif opt.cr == 30:
            level1_iter = 32 # 0.0081
    elif opt.dataset == 'bsd500_patch':
        if opt.cr == 5:
            level1_iter = 26 # 0.0264
        elif opt.cr == 10:
            level1_iter = 32 # 0.0290
        elif opt.cr == 20:
            level1_iter = 40 # 0.0348
        elif opt.cr == 30:
            level1_iter = 49 # 0.0374
    stage1_path = '%s/%s/cr%s/%s/model/lapnet0_gen_epoch_%d.pth' % (
        opt.outf, opt.dataset, opt.cr, opt.model, level1_iter)

    gen.load_state_dict(torch.load(stage1_path))

    criterion_mse = nn.MSELoss()

    cudnn.benchmark = True

    if opt.cuda:
        gen.cuda()
        criterion_mse.cuda()
        input = input.cuda()

        gen.eval()

    errD_fake_mse_total = 0
    elapsed_time = 0

    for idx, (data, _) in enumerate(testloader, 0):
        data_array = data.numpy()
        for i in range(opt.batch_size):
            for j in range(channels):
                if opt.cuda:
                    input[i, j, :] = torch.from_numpy(
                        sensing_matrix[j, :, :].dot(data_array[i, j].flatten())).cuda()
                else:
                    input[i, j, :] = torch.from_numpy(sensing_matrix[j, :, :].dot(data_array[i, j].flatten()))

        input_var = Variable(input, volatile=True)
        target = torch.from_numpy(data_array)
        if opt.cuda:
            target = target.cuda()
        target_var = Variable(target, volatile=True)

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.time()

        output = gen(input_var)

        torch.cuda.synchronize()
        end = time.time()

        elapsed_time += end - start

        print('Time cost for one batch: {:.02e}s'.format(end - start))
    print('Average time cost for one batch: {:.02e}s'.format(elapsed_time / len(testloader)))


def main():
    test_loader = data_loader()
    evaluation(test_loader)


if __name__ == '__main__':
    main()
