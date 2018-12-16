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
import models.lapgan_wgan_resnet as lapgan

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet', default='wgan_resnet')
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
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

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
                                                 transforms.Scale(opt.image_size),
                                                 transforms.CenterCrop(opt.image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                             ]))

        val_dataset = datasets.ImageFolder(root=opt.datapath + 'val_64x64',
                                           transform=transforms.Compose([
                                               transforms.Scale(opt.image_size),
                                               transforms.CenterCrop(opt.image_size),
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


def val(epoch, level, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3, sensing_matrix4,
        target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen, criterion_mse, y2, y3, y4):
    errD_fake_mse_total = 0
    for idx, (data, _) in enumerate(valloader, 0):
        if data.size(0) != opt.batch_size:
            continue

        data_array = data.numpy()
        for i in range(opt.batch_size):
            g4_target_temp = data_array[i]  # 1x64x64

            if level == 1:
                target[i] = torch.from_numpy(g4_target_temp[:, ::8, ::8])
            elif level == 2:
                target[i] = torch.from_numpy(g4_target_temp[:, ::4, ::4])  # 3x16x16
            elif level == 3:
                target[i] = torch.from_numpy(g4_target_temp[:, ::2, ::2])  # 3x32x32
            elif level == 4:
                target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64
            elif level == 5:
                target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64

            for j in range(channels):
                g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))
                y2[i, j, :] = torch.from_numpy(sensing_matrix2[j, :, :].dot(data_array[i, j].flatten()))
                y3[i, j, :] = torch.from_numpy(sensing_matrix3[j, :, :].dot(data_array[i, j].flatten()))
                y4[i, j, :] = torch.from_numpy(sensing_matrix4[j, :, :].dot(data_array[i, j].flatten()))

        g1_input_var = Variable(g1_input, volatile=True)
        if level == 1:
            output = lapnet1_gen(g1_input_var)
            target_var = Variable(target, volatile=True)
        elif level == 2:
            g2_input = lapnet1_gen(g1_input_var)
            output = lapnet2_gen(g2_input, g1_input_var)
            target_var = Variable(target, volatile=True)
        elif level == 3:
            g2_input = lapnet1_gen(g1_input_var)
            g3_input = lapnet2_gen(g2_input, g1_input_var)
            output = lapnet3_gen(g3_input, g1_input_var)
            target_var = Variable(target, volatile=True)
        elif level == 4:
            g2_input = lapnet1_gen(g1_input_var)
            g3_input = lapnet2_gen(g2_input, g1_input_var)
            g4_input = lapnet3_gen(g3_input, g1_input_var)
            output = lapnet4_gen(g4_input, g1_input_var)
            target_var = Variable(target, volatile=True)
        elif level == 5:
            y2_var = Variable(y2, volatile=True)
            y3_var = Variable(y3, volatile=True)
            y4_var = Variable(y4, volatile=True)

            g1_output = lapnet1_gen(g1_input_var)
            g2_output = lapnet2_gen(g1_output, y2_var)
            g3_output = lapnet3_gen(g2_output, y3_var)
            output = lapnet4_gen(g3_output, y4_var)
            target_var = Variable(target, volatile=True)

        errD_fake_mse = criterion_mse(output, target_var)
        errD_fake_mse_total += errD_fake_mse
        if idx % 20 == 0:
            print('Test: [%d][%d/%d] errG_mse: %.4f \n,' % (epoch, idx, len(valloader), errD_fake_mse.data[0]))

    print('Test: [%d] average errG_mse: %.4f,' % (epoch, errD_fake_mse_total.data[0] / len(valloader)))
    vutils.save_image(target_var.data,
                      '%s/%s/%s/image/test_l%d_real_samples_epoch_%03d.png' % (
                          opt.outf, opt.dataset, opt.model, level, epoch), normalize=True)
    vutils.save_image(output.data,
                      '%s/%s/%s/image/test_l%d_fake_samples_epoch_%03d.png' % (
                          opt.outf, opt.dataset, opt.model, level, epoch), normalize=True)


def train(epochs, trainloader, valloader):
    # Initialize variables
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    cr1 = 64
    cr2 = 32
    cr3 = 16
    cr4 = 8
    channels = sz_input[1]
    n = sz_input[2] ** 2
    m1 = n / cr1
    m2 = n / cr2
    m3 = n / cr3
    m4 = n / cr4
    img_size1 = 8
    img_size2 = 16
    img_size3 = 32
    img_size4 = 64

    sensing_matrix4 = randn(channels, m4, n)
    sensing_matrix3 = sensing_matrix4[:, :m3, :]
    sensing_matrix2 = sensing_matrix4[:, :m2, :]
    sensing_matrix1 = sensing_matrix4[:, :m1, :]

    g1_input = torch.FloatTensor(opt.batch_size, channels, m1)
    g2_input = torch.FloatTensor(opt.batch_size, channels, m2)
    g3_input = torch.FloatTensor(opt.batch_size, channels, m3)
    g4_input = torch.FloatTensor(opt.batch_size, channels, m4)

    g1_target = torch.FloatTensor(opt.batch_size, channels, img_size1, img_size1)
    g2_target = torch.FloatTensor(opt.batch_size, channels, img_size2, img_size2)
    g3_target = torch.FloatTensor(opt.batch_size, channels, img_size3, img_size3)
    g4_target = torch.FloatTensor(opt.batch_size, channels, img_size4, img_size4)

    y2 = torch.FloatTensor(opt.batch_size, channels, m2)
    y3 = torch.FloatTensor(opt.batch_size, channels, m3)
    y4 = torch.FloatTensor(opt.batch_size, channels, m4)

    label = torch.FloatTensor(opt.batch_size)

    fake_label = 0.1
    real_label = 0.9

    # Instantiate models
    lapnet1_gen = lapgan.LAPGAN_Generator_level1(channels, opt.ngpu)
    lapnet1_disc = lapgan.LAPGAN_Discriminator_level1(channels, opt.ngpu)
    lapnet2_gen = lapgan.LAPGAN_Generator_level2(channels, opt.ngpu, channels * m2)
    lapnet2_disc = lapgan.LAPGAN_Discriminator_level2(channels, opt.ngpu)
    lapnet3_gen = lapgan.LAPGAN_Generator_level3(channels, opt.ngpu, channels * m3)
    lapnet3_disc = lapgan.LAPGAN_Discriminator_level3(channels, opt.ngpu)
    lapnet4_gen = lapgan.LAPGAN_Generator_level4(channels, opt.ngpu, channels * m4)
    lapnet4_disc = lapgan.LAPGAN_Discriminator_level4(channels, opt.ngpu)

    # Weight initialization
    weights_init(lapnet1_gen, init_type='normal'), weights_init(lapnet1_disc, init_type='normal')
    weights_init(lapnet2_gen, init_type='normal'), weights_init(lapnet2_disc, init_type='normal')
    weights_init(lapnet3_gen, init_type='normal'), weights_init(lapnet3_disc, init_type='normal')
    weights_init(lapnet4_gen, init_type='normal'), weights_init(lapnet4_disc, init_type='normal')

    print(lapnet1_gen), print(lapnet1_disc)
    print(lapnet2_gen), print(lapnet2_disc)
    print(lapnet3_gen), print(lapnet3_disc)
    print(lapnet4_gen), print(lapnet4_disc)

    optimizer_lapnet1_gen = optim.Adam(lapnet1_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet1_disc = optim.Adam(lapnet1_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    optimizer_lapnet2_gen = optim.Adam(lapnet2_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet2_disc = optim.Adam(lapnet2_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    optimizer_lapnet3_gen = optim.Adam(lapnet3_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet3_disc = optim.Adam(lapnet3_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    optimizer_lapnet4_gen = optim.Adam(lapnet4_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet4_disc = optim.Adam(lapnet4_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    cudnn.benchmark = True

    if opt.cuda:
        lapnet1_gen.cuda(), lapnet1_disc.cuda()
        lapnet2_gen.cuda(), lapnet2_disc.cuda()
        lapnet3_gen.cuda(), lapnet3_disc.cuda()
        lapnet4_gen.cuda(), lapnet4_disc.cuda()

        criterion_mse.cuda(), criterion_bce.cuda()

        g1_input, g2_input, g3_input, g4_input = g1_input.cuda(), g2_input.cuda(), g3_input.cuda(), g4_input.cuda()
        g1_target, g2_target, g3_target, g4_target = g1_target.cuda(), g2_target.cuda(), g3_target.cuda(), g4_target.cuda()
        y2, y3, y4 = y2.cuda(), y3.cuda(), y4.cuda()
        label = label.cuda()

    for epoch in range(epochs):
        for idx, (data, _) in enumerate(trainloader):
            if data.size(0) != opt.batch_size:
                continue
            ############################
            # (1) Update D network
            ###########################

            for p in lapnet1_disc.parameters():
                p.requires_grad = True

            for p in lapnet1_disc.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 3x64x64
                # g3_target_temp = g4_target_temp[:, ::2, ::2]  # 3x32x32
                # g2_target_temp = g3_target_temp[:, ::2, ::2]  # 3x16x16
                g1_target_temp = g4_target_temp[:, ::8, ::8]  # 3x8x8
                g1_target[i] = torch.from_numpy(g1_target_temp)

                for j in range(channels):
                    g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))

            # Train disc1 with true images
            lapnet1_disc.zero_grad()
            if opt.cuda:
                g1_input = g1_input.cuda()
                g1_target = g1_target.cuda()
            g1_target_var = Variable(g1_target)

            errD_d1_real = lapnet1_disc(g1_target_var)
            errD_d1_real.backward()

            # Train disc1 with fake images
            g1_input_var = Variable(g1_input, volatile=True)
            g1_output = Variable(lapnet1_gen(g1_input_var).data)
            errD_d1_fake = lapnet1_disc(g1_output)
            errD_d1_fake = -errD_d1_fake
            errD_d1_fake.backward()
            errD = errD_d1_real + errD_d1_fake
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
            errD_g1_fake = lapnet1_disc(g1_output)
            errD_g1_fake_mse = criterion_mse(g1_output, g1_target_var)
            errD_g1 = opt.w_loss * errD_g1_fake.mean() + (1 - opt.w_loss) * errD_g1_fake_mse
            errD_g1.backward()
            optimizer_lapnet1_gen.step()

            if idx % opt.log_interval == 0:
                print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errD: %.4f, errG_fake: %.4f,'
                      'errG_mse: %.4f,' % (
                          1, epoch, epochs, idx, len(trainloader),
                          errD_d1_real.data[0],
                          errD_d1_fake.data[0],
                          errD.data[0],
                          errD_g1_fake[0],
                          errD_g1_fake_mse.data[0]
                      ))

        val(epoch, 1, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3,
            sensing_matrix4, g1_target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen,
            criterion_mse, y2, y3, y4)

        torch.save(lapnet1_gen.state_dict(),
                   '%s/%s/%s/model/lapnet1_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet1_disc.state_dict(),
                   '%s/%s/%s/model/lapnet1_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))


def main():
    train_loader, val_loader = data_loader()
    train(opt.epochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
