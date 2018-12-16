from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import os

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy.random import randn
from torch.nn import init
import copy
from torch.autograd import grad
import models.lapgan_wgan_wp as lapgan

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | woresnet | woresnetup', default='basic')
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='bsd500_patch')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--image-size', type=int, default=64, metavar='N',
                    help='The height / width of the input image to the network')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
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
parser.add_argument('--gpu', type=int, default=3, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.5, metavar='N.',
                    help='penalty for the mse and bce loss')
parser.add_argument('--LAMBDA', type=float, default=10.0, metavar='N.',
                    help='lambda for gradient penalty')

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
                                          transforms.Scale(opt.image_size),
                                          transforms.CenterCrop(opt.image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]))
    elif opt.dataset == 'mnist':
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Scale(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
        val_dataset = datasets.MNIST('./data', train=False,
                                       transform=transforms.Compose([
                                           transforms.Scale(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
    elif opt.dataset == 'bsd500':
        train_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/BSDS500/train-aug/',
                                       transform=transforms.Compose([
                                           transforms.Scale(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

        val_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/SISR/val/',
                                            transform=transforms.Compose([
                                                transforms.Scale(opt.image_size),
                                                transforms.CenterCrop(opt.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
    elif opt.dataset == 'bsd500_patch':
        train_dataset = datasets.ImageFolder(root=opt.datapath + 'train_64x64',
                                         transform=transforms.Compose([
                                             #                                            transforms.Scale(opt.image_size),
                                             #                                            transforms.CenterCrop(opt.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))

        val_dataset = datasets.ImageFolder(root=opt.datapath + 'val_64x64',
                                       transform=transforms.Compose([
                                           #                                          transforms.Scale(opt.image_size),
                                           #                                          transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
    elif opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Scale(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.Scale(opt.image_size),
                                                transforms.CenterCrop(opt.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    return train_loader, val_loader

def val(epoch, level, m1, m2, cr, channels, valloader, sensing_matrix_left, lapnet1_gen, criterion_mse, opt):
    g1_input = torch.FloatTensor(opt.batch_size, channels, m1, m2)
    g2_input = torch.FloatTensor(opt.batch_size, channels, m1 * 2, m2 * 2)

    g1_target = torch.FloatTensor(opt.batch_size, channels, m1, m2)
    g2_target = torch.FloatTensor(opt.batch_size, channels, m1 * 2, m2 * 2)

    errD_fake_mse_total = 0

    if opt.cuda:
        g1_input, g2_input = g1_input.cuda(), g2_input.cuda()
        g1_target, g2_target = g1_target.cuda(), g2_target.cuda()

    for idx, (data, _) in enumerate(valloader, 0):
        if data.size(0) != opt.batch_size:
            continue

        data_array = data.numpy()
        for i in range(opt.batch_size):
            g4_target_temp = data_array[i]  # 1x64x64
            g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
            g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
            g1_target_temp = g2_target_temp[:, ::2, ::2]  # 1x16x16
            g1_target[i] = torch.from_numpy(g1_target_temp)     # 1x8x8

            for j in range(channels):
                g1_input[i, j, :, :] = torch.from_numpy(
                    np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))

        g1_input_var = Variable(g1_input, volatile=True)
        output = lapnet1_gen(g1_input_var)
        target_var = Variable(g1_target, volatile=True)

        val_loss = criterion_mse(output, target_var)
        errD_fake_mse_total += val_loss
        print('Test: [%d/%d] errG_mse: %.4f,' % (idx, len(valloader), val_loss.data[0]))

    errD_fake_mse = errD_fake_mse_total / len(valloader)
    print('------------------------------------------')
    print('Test: [%d] errG_mse: %.4f,' % (epoch, errD_fake_mse.data[0]))
    print('------------------------------------------')
    vutils.save_image(g1_target,
                      '%s/%s/%s/image/test_l%d_real_samples_epoch_%03d.png' % (
                      opt.outf, opt.dataset, opt.model, level, epoch), normalize=True)
    vutils.save_image(output.data,
                      '%s/%s/%s/image/test_l%d_fake_samples_epoch_%03d.png.png' % (
                      opt.outf, opt.dataset, opt.model, level, epoch), normalize=True)


train_loader, val_loader = data_loader()
# Initialize variables
input, _ = train_loader.__iter__().__next__()
input = input.numpy()
sz_input = input.shape
cr = 8
channels = sz_input[1]
n1 = sz_input[2]
m1 = n1 / cr
n2 = sz_input[3]
m2 = n2 / cr

n = sz_input[2] * sz_input[3]
m = n / cr ** 2

sensing_matrix_left = randn(m, n)

g1_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1, m2)

g1_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1, m2)

label = torch.FloatTensor(opt.batch_size)

# Instantiate models
lapnet1_gen = lapgan.LAPGAN_Generator_level1(channels, opt.ngpu)
lapnet1_disc = lapgan.LAPGAN_Discriminator_level1(channels, opt.ngpu)

# Weight initialization
lapnet1_gen.apply(weights_init), lapnet1_disc.apply(weights_init)

print(lapnet1_gen), print(lapnet1_disc)

optimizer_lapnet1_gen = optim.Adam(lapnet1_gen.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizer_lapnet1_disc = optim.Adam(lapnet1_disc.parameters(), lr=opt.lr, betas=(0.5, 0.9))

#criterion_mse = nn.MSELoss()
criterion_mse = nn.L1Loss()
criterion_bce = nn.BCELoss()

one = torch.FloatTensor([1])
mone = one * -1
if opt.cuda:
    g1_input = g1_input.cuda()
    g1_target = g1_target.cuda()

    one = one.cuda()
    mone = mone.cuda()

    lapnet1_gen.cuda(), lapnet1_disc.cuda()

    criterion_mse.cuda()
    criterion_bce.cuda()

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(opt.batch_size, real_data.nelement()/opt.batch_size).contiguous().view(opt.batch_size, 3, 8, 8)
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.LAMBDA
    return gradient_penalty

def train(epochs):
    for epoch in range(epochs):
        for idx, (data, _) in enumerate(train_loader):
            if data.size(0) != opt.batch_size:
                continue
            ############################
            # (1) Update D network
            ###########################
            for p in lapnet1_disc.parameters():
                p.requires_grad = True

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 3x64x64
                g3_target_temp = g4_target_temp[:, ::2, ::2]  # 3x32x32
                g2_target_temp = g3_target_temp[:, ::2, ::2]  # 3x16x16
                g1_target_temp = g2_target_temp[:, ::2, ::2]  # 3x8x8
                g1_target[i] = torch.from_numpy(g1_target_temp)
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))

            # Train disc1 with true images
            lapnet1_disc.zero_grad()
            g1_target_var = Variable(g1_target)
            d1_real = lapnet1_disc(g1_target_var)
            d1_real = d1_real.mean()
            d1_real = -d1_real
            d1_real.backward()

            # Train disc1 with fake images
            g1_input_var = Variable(g1_input, volatile=True)
            g1_output = Variable(lapnet1_gen(g1_input_var).data)
            d1_fake = lapnet1_disc(g1_output)
            d1_fake = d1_fake.mean()
            d1_fake.backward()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(lapnet1_disc, g1_target_var.data, g1_output.data)
            gradient_penalty.backward()
            errD_d1 = d1_fake - d1_real + gradient_penalty
            optimizer_lapnet1_disc.step()

            ############################
            # (2) Update G network
            ###########################
            for p in lapnet1_disc.parameters():
                p.requires_grad = False

            # Train gen1 with fake images, disc1 is not updated
            lapnet1_gen.zero_grad()
            g1_input_var = Variable(g1_input)
            g1_output = lapnet1_gen(g1_input_var)
            g_d1_fake = lapnet1_disc(g1_output)
            errD_g1_fake_mse = criterion_mse(g1_output, g1_target_var)
            g_d1_fake = -opt.w_loss * g_d1_fake.mean() + (1 - opt.w_loss) * errD_g1_fake_mse
            g_d1_fake.backward()
            optimizer_lapnet1_gen.step()

            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errD: %.4f, gp: %.4f, '
                  'errG_fake: %.4f, errG_mse: %.4f,' % (
                      1, epoch, epochs, idx, len(train_loader),
                      d1_real.data[0],
                      d1_fake.data[0],
                      errD_d1.data[0],
                      gradient_penalty.data[0],
                      g_d1_fake.data[0],
                      errD_g1_fake_mse.data[0]
            ))

        val(epoch, 1, m1, m2, cr, channels, val_loader, sensing_matrix_left, lapnet1_gen, criterion_mse, opt)
        torch.save(lapnet1_gen.state_dict(),
                   '%s/%s/%s/model/lapnet1_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet1_disc.state_dict(),
                   '%s/%s/%s/model/lapnet1_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        vutils.save_image(g1_target,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 1, epoch),
                          normalize=True)
        vutils.save_image(g1_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 1, epoch),
                          normalize=True)

def main():
    train(opt.epochs)

if __name__ == '__main__':
    main()
