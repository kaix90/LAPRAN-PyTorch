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

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet',
                    default='reconnet')
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='cifar10')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--image-size', type=int, default=64, metavar='N',
                    help='The height / width of the input image to the network')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
parser.add_argument('--gpu', type=int, default=1, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.01, metavar='N.',
                    help='penalty for the mse and bce loss')
parser.add_argument('--cr', type=int, default=10, help='compression ratio')

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

if not os.path.exists('%s/%s/cr%s/%s/model' % (opt.outf, opt.dataset, opt.cr, opt.model)):
    os.makedirs('%s/%s/cr%s/%s/model' % (opt.outf, opt.dataset, opt.cr, opt.model))
if not os.path.exists('%s/%s/cr%s/%s/image' % (opt.outf, opt.dataset, opt.cr, opt.model)):
    os.makedirs('%s/%s/cr%s/%s/image' % (opt.outf, opt.dataset, opt.cr, opt.model))


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
    kwopt = {'num_workers': 2, 'pin_memory': True} if opt.cuda else {}

    if opt.dataset == 'lsun':
        train_dataset = datasets.LSUN(db_path=opt.datapath + 'train/', classes=['bedroom_train'],
                                      transform=transforms.Compose([
                                          transforms.Resize(opt.image_size),
                                          transforms.CenterCrop(opt.image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]))
    elif opt.dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           #transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        val_dataset = datasets.MNIST('./data', train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.image_size),
                                         transforms.CenterCrop(opt.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         #transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
    elif opt.dataset == 'bsd500':
        train_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/BSDS500/train-aug/',
                                             transform=transforms.Compose([
                                                 transforms.Resize(opt.image_size),
                                                 transforms.CenterCrop(opt.image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ]))

        val_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/SISR/val/',
                                           transform=transforms.Compose([
                                               transforms.Resize(opt.image_size),
                                               transforms.CenterCrop(opt.image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    elif opt.dataset == 'bsd500_patch':
        train_dataset = datasets.ImageFolder(root=opt.datapath + 'train_64x64',
                                             transform=transforms.Compose([
                                                 transforms.Resize(opt.image_size),
                                                 transforms.CenterCrop(opt.image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ]))

        val_dataset = datasets.ImageFolder(root=opt.datapath + 'val_64x64',
                                           transform=transforms.Compose([
                                               transforms.Resize(opt.image_size),
                                               transforms.CenterCrop(opt.image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
    elif opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(opt.image_size),
                                             transforms.CenterCrop(opt.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))

        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    return train_loader, val_loader


class net(nn.Module):
    def __init__(self, channels, leny):
        super(net, self).__init__()

        self.channels = channels
        self.base = 64
        self.fs = 64
        self.leny = leny

        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(self.channels * self.leny, self.channels * self.fs ** 2)
        self.conv1 = nn.Conv2d(self.channels, self.base, 11, 1, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base)
        self.conv2 = nn.Conv2d(self.base, self.base / 2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.base / 2)
        self.conv3 = nn.Conv2d(self.base / 2, self.channels, 7, 1, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channels)
        self.conv4 = nn.Conv2d(self.channels, self.base, 11, 1, 5, bias=False)
        self.bn4 = nn.BatchNorm2d(self.base)
        self.conv5 = nn.ConvTranspose2d(self.base, self.base / 2, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(self.base / 2)
        self.conv6 = nn.ConvTranspose2d(self.base / 2, self.channels, 7, 1, 3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        self.output = input.view(input.size(0), -1)
        self.output = self.linear1(self.output)
        self.output = self.output.view(-1, self.channels, self.fs, self.fs)
        self.output = self.relu(self.bn1(self.conv1(self.output)))
        self.output = self.relu(self.bn2(self.conv2(self.output)))
        self.output = self.relu(self.bn3(self.conv3(self.output)))
        self.output = self.relu(self.bn4(self.conv4(self.output)))
        self.output = self.relu(self.bn5(self.conv5(self.output)))
        self.output = self.conv6(self.output)
        self.output = self.tanh(self.output)

        return self.output


def val(epoch, channels, valloader, sensing_matrix, input, net, criterion_mse):
    errD_fake_mse_total = 0
    for idx, (data, _) in enumerate(valloader, 0):
        if data.size(0) != opt.batch_size:
            continue

        data_array = data.numpy()
        target = torch.from_numpy(data_array)  # 3x64x64
        if opt.cuda:
            target = target.cuda()

        for i in range(opt.batch_size):
            for j in range(channels):
                input[i, j, :] = torch.from_numpy(sensing_matrix[j, :, :].dot(data_array[i, j].flatten()))

        input_var = Variable(input, volatile=True)
        output = net(input_var)
        target_var = Variable(target, volatile=True)

        errD_fake_mse = criterion_mse(output, target_var)
        errD_fake_mse_total += errD_fake_mse
        if idx % 20 == 0:
            print('Test: [%d][%d/%d] errG_mse: %.4f \n,' % (epoch, idx, len(valloader), errD_fake_mse.data[0]))

    print('Test: [%d] average errG_mse: %.4f,' % (epoch, errD_fake_mse_total.data[0] / len(valloader)))
    vutils.save_image(target_var.data,
                      '%s/%s/cr%s/%s/image/epoch_%03d_real.png'
                      % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)
    vutils.save_image(output.data,
                      '%s/%s/cr%s/%s/image/epoch_%03d_fake.png'
                      % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)


def train(epochs, trainloader, valloader):
    # Initialize variables
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    channels = sz_input[1]
    img_size = sz_input[2]
    n = img_size ** 2
    m = n / opt.cr

    if os.path.exists('sensing_matrix_cr%d.npy' % (opt.cr)):
        sensing_matrix = np.load('sensing_matrix_cr%d.npy' % (opt.cr))
    else:
        sensing_matrix = randn(channels, m, n)

    input = torch.FloatTensor(opt.batch_size, channels, m)
    target = torch.FloatTensor(opt.batch_size, channels, img_size, img_size)

    # Instantiate models
    reconnet = net(channels, m)

    # Weight initialization
    weights_init(reconnet, init_type='normal')
    optimizer_net = optim.Adam(reconnet.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    cudnn.benchmark = True

    if opt.cuda:
        reconnet.cuda()

        criterion_mse.cuda(), criterion_bce.cuda()

        input = input.cuda()

    for epoch in range(epochs):
        # training level 0
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            reconnet.train()
            data_array = data.numpy()
            for i in range(opt.batch_size):
                for j in range(channels):
                    if opt.cuda:
                        input[i, j, :] = torch.from_numpy(
                            sensing_matrix[j, :, :].dot(data_array[i, j].flatten())).cuda()
                    else:
                        input[i, j, :] = torch.from_numpy(sensing_matrix[j, :, :].dot(data_array[i, j].flatten()))

            input_var = Variable(input)
            target = torch.from_numpy(data_array)
            if opt.cuda:
                target = target.cuda()
            target_var = Variable(target)

            # Train network
            reconnet.zero_grad()
            output = reconnet(input_var)

            err_mse = criterion_mse(output, target_var)
            err_mse.backward()
            optimizer_net.step()

            if idx % opt.log_interval == 0:
                print('[%d/%d][%d/%d] errG_mse: %.4f' % (
                    epoch, epochs, idx, len(trainloader), err_mse.data[0]))

        torch.save(reconnet.state_dict(),
                   '%s/%s/cr%s/%s/model/lapnet0_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.cr, opt.model, epoch))

        vutils.save_image(target_var.data,
                          '%s/%s/cr%s/%s/image/epoch_%03d_real.png'
                          % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)
        vutils.save_image(output.data,
                          '%s/%s/cr%s/%s/image/epoch_%03d_fake.png'
                          % (opt.outf, opt.dataset, opt.cr, opt.model, epoch), normalize=True)
        reconnet.eval()
        val(epoch, channels, valloader, sensing_matrix, input, reconnet, criterion_mse)


def main():
    train_loader, val_loader = data_loader()
    train(opt.epochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
