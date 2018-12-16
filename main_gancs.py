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
                    default='csgm')
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='cifar10')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
parser.add_argument('--gpu', type=int, default=1, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.01, metavar='N.',
                    help='penalty for the mse and bce loss')

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

if not os.path.exists('%s/%s/%s/model' % (opt.outf, opt.dataset, opt.model)):
    os.makedirs('%s/%s/%s/model' % (opt.outf, opt.dataset, opt.model))
if not os.path.exists('%s/%s/%s/image' % (opt.outf, opt.dataset, opt.model)):
    os.makedirs('%s/%s/%s/image' % (opt.outf, opt.dataset, opt.model))


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
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        val_dataset = datasets.MNIST('./data', train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.image_size),
                                         transforms.CenterCrop(opt.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
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


class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, numLayers=2):
        super(ResBlock, self).__init__()

        block = nn.Sequential(
            nn.BatchNorm2d(nFeat),
            nn.ReLU(inplace=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, stride=1, bias=False),
        )

        self.main = []
        for _ in numLayers:
            self.main.append(block)

    def forward(self, x):
        self.res = self.main(x)
        self.res += x
        self.res = self.relu(self.res)

        return self.res


class netG(nn.Module):
    def __init__(self, channels, leny):
        super(netG, self).__init__()

        self.channels = channels
        self.base = 128
        self.fs = 4
        self.leny = leny

        self.resblk1_1, self.resblk1_2, self.resblk1_3, self.resblk1_4 = ResBlock(self.base), ResBlock(self.base), \
                                                                         ResBlock(self.base), ResBlock(self.base)
        self.bn1 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(self.base, self.base, 3, 1)

        self.resblk2_1, self.resblk2_2, self.resblk1_3, self.resblk1_4 = ResBlock(self.base * 2), ResBlock(self.base * 2), \
                                                                         ResBlock(self.base * 2), ResBlock(self.base * 2)
        self.bn2 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(self.base, self.base, 3, 1)

        self.resblk3_1, self.resblk3_2, self.resblk3_3, self.resblk3_4 = ResBlock(self.base * 4), ResBlock(self.base * 4), \
                                                                         ResBlock(self.base * 4), ResBlock(self.base * 4)
        self.bn3 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(self.base, self.base, 3, 1)

        self.resblk4_1, self.resblk4_2, self.resblk4_3, self.resblk4_4 = ResBlock(self.base * 8), ResBlock(self.base * 8), \
                                                                         ResBlock(self.base * 8), ResBlock(self.base * 8)
        self.bn4 = nn.BatchNorm2d(self.base)
        self.relu = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(self.base, self.base, 3, 1)

        self.conv5 = nn.Conv2d(self.base, self.base, 3, 1)
        self.conv6 = nn.Conv2d(self.base, self.base, 1, 1)
        self.conv7 = nn.Conv2d(self.base, self.channels, 1, 1)

    def forward(self, input):
        self.output = self.relu(self.bn1(self.resblk1_1(self.resblk1_2(input))))
        self.output = self.relu(self.bn2(self.resblk2_1(self.resblk2_2(self.output))))
        self.output = self.relu(self.bn3(self.resblk3_1(self.resblk3_2(self.output))))
        self.output = self.relu(self.bn4(self.resblk4_1(self.resblk4_2(self.output))))

        self.output = self.relu(self.conv5(self.output))
        self.output = self.relu(self.conv6(self.output))
        self.output = self.conv7(self.output)

        return self.output


class netD(nn.Module):
    def __init__(self, channels):
        super(netD, self).__init__()

        self.channels = channels
        self.base = 8

        self.conv1 = nn.Conv2d(self.channels, self.base, 3, 2, bias=False)  # 64 x 32 x 32
        self.bn1 = nn.BatchNorm2d(self.base)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(self.base, self.base * 2, 3, 2, bias=False)  # 128 x 16 x 16
        self.bn2 = nn.BatchNorm2d(self.base * 2)

        self.conv3 = nn.Conv2d(self.base * 2, self.base * 4, 3, 2, bias=False)  # 256 x 8 x 8
        self.bn3 = nn.BatchNorm2d(self.base * 4)

        self.conv4 = nn.Conv2d(self.base * 4, self.base * 8, 3, 2, bias=False)  # 512 x 4 x 4
        self.bn4 = nn.BatchNorm2d(self.base * 8)

        self.conv5 = nn.Conv2d(self.base * 8, self.base * 8, 4, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.base * 8)

        self.conv6 = nn.Conv2d(self.base * 8, self.base * 8, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.base * 8)

        self.conv7 = nn.Conv2d(self.base * 8, self.base * 8, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(self.base * 8)

        self.conv8 = nn.Conv2d(self.base * 8, self.base * 8, 1, 1, bias=False)

        self.linear1 = nn.Linear(self.base * 8 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        self.output = self.lrelu(self.bn1(self.conv1(input)))
        self.output = self.lrelu(self.bn2(self.conv2(self.output)))
        self.output = self.lrelu(self.bn3(self.conv3(self.output)))
        self.output = self.lrelu(self.bn4(self.conv4(self.output)))
        self.output = self.lrelu(self.bn5(self.conv5(self.output)))
        self.output = self.lrelu(self.bn6(self.conv6(self.output)))
        self.output = self.lrelu(self.bn7(self.conv7(self.output)))
        self.output = self.conv8(self.output)
        self.output = self.output.view(self.output.size(0), -1)
        self.output = self.sigmoid(self.output)

        return self.output


def val(epoch, channels, valloader, sensing_matrix, input, gen, criterion_mse):
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
        output = gen(input_var)
        target_var = Variable(target, volatile=True)

        errD_fake_mse = criterion_mse(output, target_var)
        errD_fake_mse_total += errD_fake_mse
        if idx % 20 == 0:
            print('Test: [%d][%d/%d] errG_mse: %.4f \n,' % (epoch, idx, len(valloader), errD_fake_mse.data[0]))

    print('Test: [%d] average errG_mse: %.4f,' % (epoch, errD_fake_mse_total.data[0] / len(valloader)))
    vutils.save_image(target_var.data,
                      '%s/%s/%s/image/test_real_samples_epoch_%03d.png' % (
                          opt.outf, opt.dataset, opt.model, epoch), normalize=True)
    vutils.save_image(output.data,
                      '%s/%s/%s/image/test_fake_samples_epoch_%03d.png' % (
                          opt.outf, opt.dataset, opt.model, epoch), normalize=True)


def train(epochs, trainloader, valloader):
    # Initialize variables
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    cr = 8
    channels = sz_input[1]
    img_size = sz_input[2]
    n = img_size ** 2
    m = n / cr

    sensing_matrix = randn(channels, m, n)

    input = torch.FloatTensor(opt.batch_size, channels, m)
    target = torch.FloatTensor(opt.batch_size, channels, img_size, img_size)
    label = torch.FloatTensor(opt.batch_size)

    fake_label = 0.1
    real_label = 0.9

    # Instantiate models
    gen = netG(channels, m)
    disc = netD(channels)

    # Weight initialization
    weights_init(gen, init_type='normal'), weights_init(disc, init_type='normal')

    optimizer_lapnet_gen = optim.Adam(gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet_disc = optim.Adam(disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    cudnn.benchmark = True

    if opt.cuda:
        gen.cuda(), disc.cuda()

        criterion_mse.cuda(), criterion_bce.cuda()

        input = input.cuda()
        label = label.cuda()

    for epoch in range(epochs):
        # training level 0
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

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

            # Train disc1 with true images
            disc.zero_grad()
            d_output = disc(target_var)
            d_label_var = Variable(label.fill_(real_label))
            errD_d_real_bce = criterion_bce(d_output, d_label_var)
            errD_d_real_bce.backward()
            d_real_mean = d_output.data.mean()

            # Train disc1 with fake images
            g_output = gen(input_var)
            d_output = disc(g_output.detach())
            d_label_var = Variable(label.fill_(fake_label))
            errD_d_fake_bce = criterion_bce(d_output, d_label_var)
            errD_d_fake_bce.backward()
            optimizer_lapnet_disc.step()

            # Train gen1 with fake images
            gen.zero_grad()
            d_label_var = Variable(label.fill_(real_label))
            d_output = disc(g_output)
            errD_g_fake_bce = criterion_bce(d_output, d_label_var)
            errD_g_fake_mse = criterion_mse(g_output, target_var)
            errD_g = opt.w_loss * errD_g_fake_bce + (1 - opt.w_loss) * errD_g_fake_mse
            errD_g.backward()
            optimizer_lapnet_gen.step()
            d_fake_mean = d_output.data.mean()

            if idx % opt.log_interval == 0:
                print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                      'D(x): %.4f, D(G(z)): %.4f' % (
                          0, epoch, epochs, idx, len(trainloader),
                          errD_d_real_bce.data[0],
                          errD_d_fake_bce.data[0],
                          errD_g_fake_bce.data[0],
                          errD_g_fake_mse.data[0],
                          d_real_mean,
                          d_fake_mean))

        val(epoch, channels, valloader, sensing_matrix, input, gen, criterion_mse)

        torch.save(gen.state_dict(),
                   '%s/%s/%s/model/lapnet0_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(disc.state_dict(),
                   '%s/%s/%s/model/lapnet0_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        vutils.save_image(target,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (
                          opt.outf, opt.dataset, opt.model, 2, epoch),
                          normalize=True)
        vutils.save_image(g_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (
                          opt.outf, opt.dataset, opt.model, 2, epoch),
                          normalize=True)


def main():
    train_loader, val_loader = data_loader()
    train(opt.epochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
