from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2
import models.lapgan_mnist as lapgan
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import os

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy.random import randn
import copy

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | woresnet | woresnetup', default='basic')
parser.add_argument('--dataset', help='lsun | imagenet | mnist', default='bsd500')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--layers-gan', type=int, default=3, metavar='N',
                    help='number of hierarchies in the GAN (default: 64)')
parser.add_argument('--gpu', type=int, default=2, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.001, metavar='N.',
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
        train_dataset = datasets.ImageFolder(root=opt.datapath + 'train_64x64/',
                                       transform=transforms.Compose([
                                           transforms.Scale(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

        val_dataset = datasets.ImageFolder(root=opt.datapath + 'val_64x64/',
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

def val(epoch, level, m1, m2, channels, valloader, sensing_matrix_left, target, g1_input, lapnet1, lapnet2_gen,
        lapnet3_gen, lapnet4_gen, lapnet_gen, criterion_mse):
    errD_fake_mse_total = 0
    for idx, (data, _) in enumerate(valloader, 0):
        if data.size(0) != opt.batch_size:
            continue

        data_array = data.numpy()
        for i in range(opt.batch_size):
            g4_target_temp = data_array[i]  # 1x64x64
            g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
            g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
            if level == 2:
                target[i] = torch.from_numpy(g2_target_temp)     # 3x16x16
            if level == 3:
                target[i] = torch.from_numpy(g3_target_temp)     # 3x32x32
            if level == 4:
                target[i] = torch.from_numpy(g4_target_temp)     # 3x64x64
            elif level == 5:
                target[i] = torch.from_numpy(g4_target_temp)     # 3x64x64

            for j in range(channels):
                g1_input[i, j, :, :] = torch.from_numpy(
                    np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))

        g1_input_var = Variable(g1_input, volatile=True)
        if level == 2:
            g2_input = lapnet1(g1_input_var)
            output = lapnet2_gen(g2_input, g1_input_var)
            target_var = Variable(target, volatile=True)
        if level == 3:
            g2_input = lapnet1(g1_input_var)
            g3_input = lapnet2_gen(g2_input, g1_input_var)
            output = lapnet3_gen(g3_input, g1_input_var)
            target_var = Variable(target, volatile=True)
        elif level == 4:
            g2_input = lapnet1(g1_input_var)
            g3_input = lapnet2_gen(g2_input, g1_input_var)
            g4_input = lapnet3_gen(g3_input, g1_input_var)
            output = lapnet4_gen(g4_input, g1_input_var)
            target_var = Variable(target, volatile=True)
        elif level == 5:
            output = lapnet_gen(g1_input_var, g1_input_var)
            target_var = Variable(target, volatile=True)

        errD_fake_mse_total += criterion_mse(output, target_var)

    errD_fake_mse = errD_fake_mse_total / len(valloader)
    print('Test: [%d] errG_mse: %.4f,' % (epoch, errD_fake_mse.data[0]))
    vutils.save_image(target,
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
    g2_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1 * 2, m2 * 2)
    g3_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1 * 4, m2 * 4)
    g4_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1 * 8, m2 * 8)

    g1_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1, m2)
    g2_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1 * 2, m2 * 2)
    g3_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1 * 4, m2 * 4)
    g4_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1 * 8, m2 * 8)

    label = torch.FloatTensor(opt.batch_size)

    fake_label = 0.1
    real_label = 0.9

    # Instantiate models
    lapnet1 = lapgan.LAPGAN_Generator_level1(channels, opt.ngpu)
    lapnet2_gen = lapgan.LAPGAN_Generator_level2(channels, opt.ngpu, channels * m1 * m2)
    lapnet2_disc = lapgan.LAPGAN_Discriminator_level2(channels, opt.ngpu)
    lapnet3_gen = lapgan.LAPGAN_Generator_level3(channels, opt.ngpu, channels * m1 * m2)
    lapnet3_disc = lapgan.LAPGAN_Discriminator_level3(channels, opt.ngpu)
    lapnet4_gen = lapgan.LAPGAN_Generator_level4(channels, opt.ngpu, channels * m1 * m2)
    lapnet4_disc = lapgan.LAPGAN_Discriminator_level4(channels, opt.ngpu)

    lapnet_gen = lapgan.LAPGAN(channels, opt.ngpu, lapnet1, lapnet2_gen, lapnet3_gen, lapnet4_gen)
    optimizer_lapnet_gen = optim.Adam(lapnet_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Weight initialization
    lapnet1.apply(weights_init)
    lapnet2_disc.apply(weights_init)
    lapnet2_gen.apply(weights_init)
    lapnet3_disc.apply(weights_init)
    lapnet3_gen.apply(weights_init)
    lapnet4_disc.apply(weights_init)
    lapnet4_gen.apply(weights_init)

    print(lapnet1)
    print(lapnet2_disc)
    print(lapnet2_gen)
    print(lapnet3_disc)
    print(lapnet3_gen)
    print(lapnet4_disc)
    print(lapnet4_gen)

    optimizer_lapnet1 = optim.Adam(lapnet1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet2_disc = optim.Adam(lapnet2_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet2_gen = optim.Adam(lapnet2_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet3_disc = optim.Adam(lapnet3_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet3_gen = optim.Adam(lapnet3_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet4_disc = optim.Adam(lapnet4_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet4_gen = optim.Adam(lapnet4_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    #    torch.cuda.set_device(gpus[0])
    if opt.gpu:
        lapnet1.cuda()
        lapnet2_gen.cuda()
        lapnet2_disc.cuda()
        lapnet3_gen.cuda()
        lapnet3_disc.cuda()
        lapnet4_gen.cuda()
        lapnet4_disc.cuda()

        criterion_mse.cuda()
        criterion_bce.cuda()

        g1_input, g2_input, g3_input, g4_input = g1_input.cuda(), g2_input.cuda(), g3_input.cuda(), g4_input.cuda()
        g1_target, g2_target, g3_target, g4_target = g1_target.cuda(), g2_target.cuda(), g3_target.cuda(), g4_target.cuda()
        label = label.cuda()

    for epoch in range(epochs):
        # training level 1

        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 1x64x64
                g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
                g1_target_temp = g2_target_temp[:, ::2, ::2]  # 1x8x8
                g2_target[i] = torch.from_numpy(g2_target_temp)
                g1_target[i] = torch.from_numpy(g1_target_temp)

                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))

            g1_input_var = Variable(g1_input)
            target_var = Variable(g1_target)

            lapnet1.zero_grad()
            outputs = lapnet1(g1_input_var)
            loss = criterion_mse(outputs, target_var)
            loss.backward()
            optimizer_lapnet1.step()
            torch.save(lapnet1.state_dict(),
                       '%s/%s/%s/model/lapnet1_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
            print('Level %d [%d/%d][%d/%d] loss_G: %.4f' % (1, epoch, epochs, idx, len(trainloader), loss.data[0]))


        # training level 2
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 1x64x64
                g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
                g2_target[i] = torch.from_numpy(g2_target_temp)
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))

            g1_input_var = Variable(g1_input)
            g2_input = lapnet1(g1_input_var)

            # Train disc2 with true images
            g2_target_var = Variable(g2_target)
            lapnet2_disc.zero_grad()
            d2_output = lapnet2_disc(g2_target_var)
            d2_label_var = Variable(label.fill_(real_label))
            errD_d2_real_bce = criterion_bce(d2_output, d2_label_var)
            errD_d2_real_bce.backward()
            d2_real_mean = d2_output.data.mean()

            # Train disc2 with fake images
            g2_output = lapnet2_gen(g2_input, g1_input_var)
            d2_output = lapnet2_disc(g2_output.detach())
            d2_label_var = Variable(label.fill_(fake_label))
            errD_d2_fake_bce = criterion_bce(d2_output, d2_label_var)
            errD_d2_fake_bce.backward()
            optimizer_lapnet2_disc.step()

            # Train gen2 with fake images, disc2 is not updated
            lapnet2_gen.zero_grad()
            d2_label_var = Variable(label.fill_(real_label))
            d2_output = lapnet2_disc(g2_output)
            errD_g2_fake_bce = criterion_bce(d2_output, d2_label_var)
            errD_g2_fake_mse = criterion_mse(g2_output, g2_target_var)
            errD_g2 = opt.w_loss * errD_g2_fake_bce + (1 - opt.w_loss) * errD_g2_fake_mse
            errD_g2.backward()
            optimizer_lapnet2_gen.step()
            d2_fake_mean = d2_output.data.mean()

            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                  'D(x): %.4f, D(G(z)): %.4f' % (
                      2, epoch, epochs, idx, len(trainloader),
                      errD_d2_real_bce.data[0],
                      errD_d2_fake_bce.data[0],
                      errD_g2_fake_bce.data[0],
                      errD_g2_fake_mse.data[0],
                      d2_real_mean,
                      d2_fake_mean))

        val(epoch, 2, m1, m2, channels, valloader, sensing_matrix_left, g2_target, g1_input, lapnet1,
            lapnet2_gen, lapnet3_gen, lapnet4_gen, lapnet_gen, criterion_mse)
        torch.save(lapnet2_gen.state_dict(),
                   '%s/%s/%s/model/lapnet2_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet2_disc.state_dict(),
                   '%s/%s/%s/model/lapnet2_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        vutils.save_image(g2_target,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 2, epoch),
                          normalize=True)
        vutils.save_image(g2_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 2, epoch),
                          normalize=True)

        # training level 3
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 1x64x64
                g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                g3_target[i] = torch.from_numpy(g3_target_temp)
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))
            g1_input_var = Variable(g1_input)
            g2_input = lapnet1(g1_input_var)  # 1x8x8
            g3_input = lapnet2_gen(g2_input, g1_input_var)  # 1x16x16

            # Train disc3 with true images
            g3_target_var = Variable(g3_target)
            lapnet3_disc.zero_grad()
            d3_output = lapnet3_disc(g3_target_var)
            d3_label_var = Variable(label.fill_(real_label))
            errD_d3_real_bce = criterion_bce(d3_output, d3_label_var)
            errD_d3_real_bce.backward()
            d3_real_mean = d3_output.data.mean()
            # Train disc3 with fake images
            g3_output = lapnet3_gen(g3_input, g1_input_var)
            d3_output = lapnet3_disc(g3_output.detach())
            d3_label_var = Variable(label.fill_(fake_label))
            errD_d3_fake_bce = criterion_bce(d3_output, d3_label_var)
            errD_d3_fake_bce.backward()
            optimizer_lapnet3_disc.step()
            # Train gen3 with fake images, disc3 is not updated
            lapnet3_gen.zero_grad()
            d3_label_var = Variable(label.fill_(real_label))
            d3_output = lapnet3_disc(g3_output)
            errD_g3_fake_bce = criterion_bce(d3_output, d3_label_var)
            errD_g3_fake_mse = criterion_mse(g3_output, g3_target_var)
            errD_g3 = opt.w_loss * errD_g3_fake_bce + (1 - opt.w_loss) * errD_g3_fake_mse
            errD_g3.backward()
            optimizer_lapnet3_gen.step()
            d3_fake_mean = d3_output.data.mean()

            lapnet3_gen.conv2.weight = copy.deepcopy(lapnet2_gen.conv2.weight)
            lapnet3_gen.conv3.weight = copy.deepcopy(lapnet2_gen.conv3.weight)
            lapnet3_gen.deconv6.weigth = copy.deepcopy(lapnet2_gen.deconv4.weight)

            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                  'D(x): %.4f, D(G(z)): %.4f' % (
                      3, epoch, epochs, idx, len(trainloader),
                      errD_d3_real_bce.data[0],
                      errD_d3_fake_bce.data[0],
                      errD_g3_fake_bce.data[0],
                      errD_g3_fake_mse.data[0],
                      d3_real_mean,
                      d3_fake_mean))

        val(epoch, 3, m1, m2, channels, valloader, sensing_matrix_left, g3_target, g1_input, lapnet1,
            lapnet2_gen, lapnet3_gen, lapnet4_gen, lapnet_gen, criterion_mse)
        torch.save(lapnet3_gen.state_dict(),
                   '%s/%s/%s/model/lapnet3_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet3_disc.state_dict(),
                   '%s/%s/%s/model/lapnet3_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        vutils.save_image(g3_target,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 3, epoch),
                          normalize=True)
        vutils.save_image(g3_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 3, epoch),
                          normalize=True)

        # training level 4
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 1x64x64
                g4_target[i] = torch.from_numpy(g4_target_temp)

                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))
            g1_input_var = Variable(g1_input)
            g2_input = lapnet1(g1_input_var)  # 1x8x8
            g3_input = lapnet2_gen(g2_input, g1_input_var)  # 1x16x16
            g4_input = lapnet3_gen(g3_input, g1_input_var)  # 1x32x32

            # Train disc4 with true images
            g4_target_var = Variable(g4_target)
            lapnet4_disc.zero_grad()
            d4_output = lapnet4_disc(g4_target_var)
            d4_label_var = Variable(label.fill_(real_label))
            errD_d4_real_bce = criterion_bce(d4_output, d4_label_var)
            errD_d4_real_bce.backward()
            d4_real_mean = d4_output.data.mean()
            # Train disc4 with fake images
            g4_output = lapnet4_gen(g4_input, g1_input_var)
            d4_output = lapnet4_disc(g4_output.detach())
            d4_label_var = Variable(label.fill_(fake_label))
            errD_d4_fake_bce = criterion_bce(d4_output, d4_label_var)
            errD_d4_fake_bce.backward()
            optimizer_lapnet4_disc.step()
            # Train gen4 with fake images, disc4 is not updated
            lapnet4_gen.zero_grad()
            d4_label_var = Variable(label.fill_(real_label))
            d4_output = lapnet4_disc(g4_output)
            errD_g4_fake_bce = criterion_bce(d4_output, d4_label_var)
            errD_g4_fake_mse = criterion_mse(g4_output, g4_target_var)
            errD_g4 = opt.w_loss * errD_g4_fake_bce + (1 - opt.w_loss) * errD_g4_fake_mse
            errD_g4.backward()
            optimizer_lapnet4_gen.step()
            d4_fake_mean = d4_output.data.mean()

            lapnet4_gen.conv2.weight = copy.deepcopy(lapnet3_gen.conv2.weight)
            lapnet4_gen.conv3.weight = copy.deepcopy(lapnet3_gen.conv3.weight)
            lapnet4_gen.conv4.weight = copy.deepcopy(lapnet3_gen.conv4.weight)

            lapnet4_gen.deconv7.weigth = copy.deepcopy(lapnet3_gen.deconv5.weight)
            lapnet4_gen.deconv8.weigth = copy.deepcopy(lapnet3_gen.deconv6.weight)

            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                  'D(x): %.4f, D(G(z)): %.4f' % (
                      4, epoch, epochs, idx, len(trainloader),
                      errD_d4_real_bce.data[0],
                      errD_d4_fake_bce.data[0],
                      errD_g4_fake_bce.data[0],
                      errD_g4_fake_mse.data[0],
                      d4_real_mean,
                      d4_fake_mean))

        val(epoch, 4, m1, m2, channels, valloader, sensing_matrix_left, g4_target, g1_input, lapnet1,
            lapnet2_gen, lapnet3_gen, lapnet4_gen, lapnet_gen, criterion_mse)
        torch.save(lapnet4_gen.state_dict(),
                   '%s/%s/%s/model/lapnet4_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet4_disc.state_dict(),
                   '%s/%s/%s/model/lapnet4_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        vutils.save_image(g4_target,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 4, epoch),
                          normalize=True)
        vutils.save_image(g4_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 4, epoch),
                          normalize=True)

        # training the whole model from all the sub-models

        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 1x64x64
                g4_target[i] = torch.from_numpy(g4_target_temp)
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        np.reshape(sensing_matrix_left.dot(data_array[i, j].flatten()), (m1, m2)))

            # Train lapnet_disc with true images
            g1_input_var = Variable(g1_input)
            g4_target_var = Variable(g4_target)
            lapnet4_disc.zero_grad()
            d4_output = lapnet4_disc(g4_target_var)
            d4_label_var = Variable(label.fill_(real_label))
            errD_d4_real_bce = criterion_bce(d4_output, d4_label_var)
            errD_d4_real_bce.backward()
            d4_real_mean = d4_output.data.mean()

            # Train lapnet_disc with fake images
            g4_output = lapnet_gen(g1_input_var, g1_input_var)
            d4_output = lapnet4_disc(g4_output.detach())
            d4_label_var = Variable(label.fill_(fake_label))
            errD_d4_fake_bce = criterion_bce(d4_output, d4_label_var)
            errD_d4_fake_bce.backward()
            optimizer_lapnet4_disc.step()

            # Train lapnet_gen with fake images, lapgen_disc is not updated
            lapnet_gen.zero_grad()
            d4_label_var = Variable(label.fill_(real_label))
            d4_output = lapnet4_disc(g4_output)
            errD_g4_fake_bce = criterion_bce(d4_output, d4_label_var)
            errD_g4_fake_mse = criterion_mse(g4_output, g4_target_var)
            errD_g4 = opt.w_loss * errD_g4_fake_bce + (1 - opt.w_loss) * errD_g4_fake_mse
            errD_g4.backward()
            optimizer_lapnet_gen.step()
            d4_fake_mean = d4_output.data.mean()

            lapnet4_gen.conv2.weight = copy.deepcopy(lapnet3_gen.conv2.weight)
            lapnet4_gen.conv3.weight = copy.deepcopy(lapnet3_gen.conv3.weight)
            lapnet4_gen.conv4.weight = copy.deepcopy(lapnet3_gen.conv4.weight)

            lapnet4_gen.deconv7.weigth = copy.deepcopy(lapnet3_gen.deconv5.weight)
            lapnet4_gen.deconv8.weigth = copy.deepcopy(lapnet3_gen.deconv6.weight)

            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                  'D(x): %.4f, D(G(z)): %.4f' % (
                      5, epoch, epochs, idx, len(trainloader),
                      errD_d4_real_bce.data[0],
                      errD_d4_fake_bce.data[0],
                      errD_g4_fake_bce.data[0],
                      errD_g4_fake_mse.data[0],
                      d4_real_mean,
                      d4_fake_mean))


        val(epoch, 5, m1, m2, channels, valloader, sensing_matrix_left, g4_target, g1_input, lapnet1,
            lapnet2_gen, lapnet3_gen, lapnet4_gen, lapnet_gen, criterion_mse)

        torch.save(lapnet_gen.state_dict(),
                   '%s/%s/%s/model/lapnet_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet4_disc.state_dict(),
                   '%s/%s/%s/model/lapnet_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        vutils.save_image(g4_target,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 5, epoch),
                          normalize=True)
        vutils.save_image(g4_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 5, epoch),
                          normalize=True)

def main():
    train_loader, val_loader = data_loader()
    train(opt.epochs, train_loader, val_loader)

if __name__ == '__main__':
    main()
