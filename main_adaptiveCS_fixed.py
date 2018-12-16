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
import string, copy
from scipy import linalg
import math

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet',
                    default='adaptiveCS_resnet_wy_ifusion_ufirst_fixed')  # adaptiveCS_resnet_wy_lfusion_ufirst
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='cifar10')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
parser.add_argument('--stage', type=int, default=1, help='the stage under training')
parser.add_argument('--transfer', action='store_true', default=False, help='transfer weights')
parser.add_argument('--cr', type=int, default=10, help='compression ratio')

opt = parser.parse_args()
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: please run with GPU")
print(opt)

if 'woy' in opt.model:
    import models.lapgan_adaptiveCS_resnet_woy as lapgan
elif 'lfusion' in opt.model:
    import models.lapgan_adaptiveCS_latefusion as lapgan
# elif 'adaptiveCS_resnet' in opt.model:
elif 'mnist' in opt.dataset:
    import models.lapgan_adaptiveCS_resnet_mnist as lapgan
elif 'bsd500_patch' in opt.dataset:
    import models.lapgan_adaptiveCS_resnet_bsd500 as lapgan
else:
    import models.lapgan_adaptiveCS_resnet as lapgan
# else:
#    import models.lapgan_adaptiveCS as lapgan

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

if not os.path.exists('%s/%s/cr%s/%s/stage%s/model' % (opt.outf, opt.dataset, opt.cr, opt.model, opt.stage)):
    os.makedirs('%s/%s/cr%s/%s/stage%s/model' % (opt.outf, opt.dataset, opt.cr, opt.model, opt.stage))
if not os.path.exists('%s/%s/cr%s/%s/stage%s/image' % (opt.outf, opt.dataset, opt.cr, opt.model, opt.stage)):
    os.makedirs('%s/%s/cr%s/%s/stage%s/image' % (opt.outf, opt.dataset, opt.cr, opt.model, opt.stage))


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
                                       ]))
        val_dataset = datasets.MNIST('./data', train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.image_size),
                                         transforms.CenterCrop(opt.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))

        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(opt.image_size),
                                           transforms.CenterCrop(opt.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    return train_loader, val_loader


def val(epoch, stage, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3, sensing_matrix4,
        target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen, criterion_mse, y2, y3, y4):
    errD_fake_mse_total = 0
    for idx, (data, _) in enumerate(valloader, 0):
        if data.size(0) != opt.batch_size:
            continue

        data_array = data.numpy()
        for i in range(opt.batch_size):
            g4_target_temp = data_array[i]  # 1x64x64

            if stage == 1:
                target[i] = torch.from_numpy(g4_target_temp[:, ::8, ::8])
            elif stage == 2:
                target[i] = torch.from_numpy(g4_target_temp[:, ::4, ::4])  # 3x16x16
            elif stage == 3:
                target[i] = torch.from_numpy(g4_target_temp[:, ::2, ::2])  # 3x32x32
            elif stage == 4:
                target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64
            elif stage == 5:
                target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64

            for j in range(channels):
                g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))
                y2[i, j, :] = torch.from_numpy(sensing_matrix2[j, :, :].dot(data_array[i, j].flatten()))
                y3[i, j, :] = torch.from_numpy(sensing_matrix3[j, :, :].dot(data_array[i, j].flatten()))
                y4[i, j, :] = torch.from_numpy(sensing_matrix4[j, :, :].dot(data_array[i, j].flatten()))

        g1_input_var = Variable(g1_input, volatile=True)
        if stage == 1:
            output = lapnet1_gen(g1_input_var)
            target_var = Variable(target, volatile=True)
        elif stage == 2:
            y2_var = Variable(y2)
            g2_input = lapnet1_gen(g1_input_var)
            output = lapnet2_gen(g2_input, y2_var)
            target_var = Variable(target, volatile=True)
        elif stage == 3:
            y2_var = Variable(y2)
            y3_var = Variable(y3)
            g2_input = lapnet1_gen(g1_input_var)
            g3_input = lapnet2_gen(g2_input, y2_var)
            output = lapnet3_gen(g3_input, y3_var)
            target_var = Variable(target, volatile=True)
        elif stage == 4:
            y2_var = Variable(y2)
            y3_var = Variable(y3)
            y4_var = Variable(y4)
            g2_input = lapnet1_gen(g1_input_var)
            g3_input = lapnet2_gen(g2_input, y2_var)
            g4_input = lapnet3_gen(g3_input, y3_var)
            output = lapnet4_gen(g4_input, y4_var)
            target_var = Variable(target, volatile=True)

        errD_fake_mse = criterion_mse(output, target_var)
        errD_fake_mse_total += errD_fake_mse
        if idx % 20 == 0:
            print('Test: [%d][%d/%d] errG_mse: %.4f \n,' % (epoch, idx, len(valloader), errD_fake_mse.data[0]))

    print('Test: [%d] average errG_mse: %.4f,' % (epoch, errD_fake_mse_total.data[0] / len(valloader)))
    vutils.save_image(target_var.data, '%s/%s/cr%s/%s/stage%s/image/test_epoch_%03d_real.png'
                      % (opt.outf, opt.dataset, opt.cr, opt.model, opt.stage, epoch), normalize=True)
    vutils.save_image(output.data, '%s/%s/cr%s/%s/stage%s/image/test_epoch_%03d_fake.png'
                      % (opt.outf, opt.dataset, opt.cr, opt.model, opt.stage, epoch), normalize=True)


def train(epochs, trainloader, valloader):
    # Initialize variables
    input, _ = trainloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    cr1 = 8 * opt.cr
    cr2 = 4 * opt.cr
    cr3 = 2 * opt.cr
    cr4 = opt.cr
    channels = sz_input[1]
    n = sz_input[2] ** 2
    '''
    m1 = int(math.ceil(float(n) / cr1))
    m2 = int(math.ceil(float(n) / cr2))
    m3 = int(math.ceil(float(n) / cr3))
    m4 = int(math.ceil(float(n) / cr4))
    '''
    m1 = n / cr4
    m2 = n / cr4
    m3 = n / cr4
    m4 = n / cr4
    img_size1 = sz_input[3] / 8
    img_size2 = sz_input[3] / 4
    img_size3 = sz_input[3] / 2
    img_size4 = sz_input[3]

    if os.path.exists('sensing_matrix_cr%d.npy' % (opt.cr)):
        sensing_matrix4 = np.load('sensing_matrix_cr%d.npy' % (opt.cr))
    else:
        sensing_matrix4 = randn(channels, m4, n)
        # sensing_matrix4 = np.zeros(sensing_matrix4_unnorm.shape)
        # for chan in range(channels):
        #    M = sensing_matrix4_unnorm[chan, :, :]
        #    M = np.transpose(linalg.orth(np.transpose(M)))
        #    sensing_matrix4[chan, :, :] = M
        np.save('sensing_matrix_cr%d.npy' % (opt.cr), sensing_matrix4)

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
    lapnet1_gen = lapgan.LAPGAN_Generator_level1(channels, channels * m1, opt.ngpu)
    lapnet1_disc = lapgan.LAPGAN_Discriminator_level1(channels, opt.ngpu)
    lapnet2_gen = lapgan.LAPGAN_Generator_level2(channels, channels * m2, opt.ngpu)
    lapnet2_disc = lapgan.LAPGAN_Discriminator_level2(channels, opt.ngpu)
    lapnet3_gen = lapgan.LAPGAN_Generator_level3(channels, channels * m3, opt.ngpu)
    lapnet3_disc = lapgan.LAPGAN_Discriminator_level3(channels, opt.ngpu)
    lapnet4_gen = lapgan.LAPGAN_Generator_level4(channels, channels * m4, opt.ngpu)
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

    if opt.dataset == 'bsd500_patch':
        if 'wy' in opt.model and 'ifusion' in opt.model:
            if opt.cr == 5:
                level1_iter = 7  # 0.0264
                level2_iter = 11  # 0.0157
                level3_iter = 12  # 0.0090
                level4_iter = 1
            elif opt.cr == 10:
                level1_iter = 16  # 0.0353 18  # 14
                level2_iter = 16  # 0.0230 16  # 27
                level3_iter = 16  # 0.0144 8  # 27
                level4_iter = 21
            elif opt.cr == 20:
                level1_iter = 6  # 0.0450
                level2_iter = 2  # 0.0389 9
                level3_iter = 1  # 0.0318
                level4_iter = 93
            elif opt.cr == 30:
                level1_iter = 5  # 0.0508
                level2_iter = 0  # 0.0420 5
                level3_iter = 0  # 0.0339 3

    if opt.dataset == 'mnist':
        if 'wy' in opt.model and 'ifusion' in opt.model:
            if opt.cr == 5:
                level1_iter = 94  # 0.0074
                level2_iter = 36  # 0.0052
                level3_iter = 88
                level4_iter = 1
            elif opt.cr == 10:
                level1_iter = 73  # 0.0193
                level2_iter = 31  # 0.0111 16  # 27
                level3_iter = 78  # 0.0077 8  # 27
                level4_iter = 96  # 0.0034
            elif opt.cr == 20:
                level1_iter = 57  # 0.0499
                level2_iter = 57  # 0.0233
                level3_iter = 95  # 0.0122
                level4_iter = 83  # 0.0059
            elif opt.cr == 30:
                level1_iter = 34  # 0.0771
                level2_iter = 66  # 0.0378 5
                level3_iter = 99  # 0.0177 3
                level4_iter = 88  # 0.0056

    if opt.dataset == 'cifar10':
        if 'wy' in opt.model and 'ifusion' in opt.model:
            if opt.cr == 5:
                level1_iter = 28  # 0.0126
                level2_iter = 81  # 0.0039
                level3_iter = 21  # 0.0025
                level4_iter = 26  # 0.0008
            elif opt.cr == 10:
                level1_iter = 24  # 0.0242
                level2_iter = 33  # 0.0102
                level3_iter = 58  # 0.0045
                level4_iter = 21  # 0.0017
            elif opt.cr == 20:
                level1_iter = 19  # 0.0420
                level2_iter = 30  # 0.0221
                level3_iter = 44  # 0.0111
                level4_iter = 21  # 0.0043
            elif opt.cr == 30:
                level1_iter = 16  # 0.0532
                level2_iter = 34  # 0.0317
                level3_iter = 84  # 0.0174
                level4_iter = 16  # 0.0078
        elif 'wy' in opt.model and 'lfusion' in opt.model:
            if opt.cr == 10:
                level1_iter = 18
                level2_iter = 16
                level3_iter = 22
                level4_iter = 99
        elif 'woy' in opt.model:
            if opt.cr == 10:
                level1_iter = 18
                level2_iter = 7
                level3_iter = 5
                level4_iter = 1

    stage1_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
        opt.outf, opt.dataset, opt.cr, opt.model, 1, level1_iter)
    stage2_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
        opt.outf, opt.dataset, opt.cr, opt.model, 2, level2_iter)
    stage3_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
        opt.outf, opt.dataset, opt.cr, opt.model, 3, level3_iter)
    stage4_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
        opt.outf, opt.dataset, opt.cr, opt.model, 4, level4_iter)

    # if opt.stage == 1:
    #  if os.path.isfile(stage1_path):
    #    lapnet1_gen.load_state_dict(stage1_path)

    if opt.stage == 2:
        lapnet1_gen.load_state_dict(torch.load(stage1_path))
        print('loading level1 iteration' + str(level1_iter))
    elif opt.stage == 3:
        lapnet1_gen.load_state_dict(torch.load(stage1_path))
        lapnet2_gen.load_state_dict(torch.load(stage2_path))
        print('loading level1 iteration' + str(level1_iter))
        print('loading level2 iteration' + str(level2_iter))
    elif opt.stage == 4:
        lapnet1_gen.load_state_dict(torch.load(stage1_path))
        lapnet2_gen.load_state_dict(torch.load(stage2_path))
        lapnet3_gen.load_state_dict(torch.load(stage3_path))
        print('loading level1 iteration' + str(level1_iter))
        print('loading level2 iteration' + str(level2_iter))
        print('loading level3 iteration' + str(level3_iter))
    elif opt.stage == 5:
        lapnet1_gen.load_state_dict(torch.load(stage1_path))
        lapnet2_gen.load_state_dict(torch.load(stage2_path))
        lapnet3_gen.load_state_dict(torch.load(stage3_path))
        lapnet4_gen.load_state_dict(torch.load(stage4_path))
        print('loading level1 iteration' + str(level1_iter))
        print('loading level2 iteration' + str(level2_iter))
        print('loading level3 iteration' + str(level3_iter))
        print('loading level4 iteration' + str(level4_iter))

    for epoch in range(epochs):
        '''
        if epoch < 2 :
            opt.w_loss = 0.1
        elif epoch  < 4:
            opt.w_loss = 0.01
        else:
            opt.w_loss = 0.001
        '''

        # training level 1
        if opt.stage == 1 or opt.stage == 5:
            for idx, (data, _) in enumerate(trainloader, 0):
                if data.size(0) != opt.batch_size:
                    continue

                lapnet1_gen.train(), lapnet1_disc.train()
                data_array = data.numpy()
                for i in range(opt.batch_size):
                    g4_target_temp = data_array[i]  # 3x64x64
                    # g3_target_temp = g4_target_temp[:, ::2, ::2]  # 3x32x32
                    # g2_target_temp = g3_target_temp[:, ::2, ::2]  # 3x16x16
                    g1_target_temp = g4_target_temp[:, ::8, ::8]  # 3x8x8
                    g1_target[i] = torch.from_numpy(g1_target_temp)

                    for j in range(channels):
                        g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))

                g1_input_var = Variable(g1_input)
                g1_target_var = Variable(g1_target)

                # Train disc1 with true images
                lapnet1_disc.zero_grad()
                d1_output = lapnet1_disc(g1_target_var)
                d1_label_var = Variable(label.fill_(real_label))
                errD_d1_real_bce = criterion_bce(d1_output, d1_label_var)
                errD_d1_real_bce.backward()
                d1_real_mean = d1_output.data.mean()

                # Train disc1 with fake images
                g1_output = lapnet1_gen(g1_input_var)
                d1_output = lapnet1_disc(g1_output.detach())
                d1_label_var = Variable(label.fill_(fake_label))
                errD_d1_fake_bce = criterion_bce(d1_output, d1_label_var)
                errD_d1_fake_bce.backward()
                optimizer_lapnet1_disc.step()

                # Train gen1 with fake images
                lapnet1_gen.zero_grad()
                d1_label_var = Variable(label.fill_(real_label))
                d1_output = lapnet1_disc(g1_output)
                errD_g1_fake_bce = criterion_bce(d1_output, d1_label_var)
                errD_g1_fake_mse = criterion_mse(g1_output, g1_target_var)
                errD_g1 = opt.w_loss * errD_g1_fake_bce + (1 - opt.w_loss) * errD_g1_fake_mse
                errD_g1.backward()
                optimizer_lapnet1_gen.step()
                d1_fake_mean = d1_output.data.mean()

                if idx % opt.log_interval == 0:
                    print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                          'D(x): %.4f, D(G(z)): %.4f' % (
                              1, epoch, epochs, idx, len(trainloader),
                              errD_d1_real_bce.data[0],
                              errD_d1_fake_bce.data[0],
                              errD_g1_fake_bce.data[0],
                              errD_g1_fake_mse.data[0],
                              d1_real_mean,
                              d1_fake_mean))

            torch.save(lapnet1_gen.state_dict(), '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 1, epoch))
            torch.save(lapnet1_disc.state_dict(), '%s/%s/cr%s/%s/stage%s/model/disc_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 1, epoch))

            vutils.save_image(g1_target_var.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 1, epoch), normalize=True)
            vutils.save_image(g1_output.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 1, epoch), normalize=True)

            lapnet1_gen.eval(), lapnet1_disc.eval()
            val(epoch, 1, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3,
                sensing_matrix4, g1_target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen,
                criterion_mse, y2, y3, y4)

        # training level 2
        # load weight of level 1
        if opt.stage == 2 or opt.stage == 5:
            print('loading level1 iteration' + str(level1_iter))
            for idx, (data, _) in enumerate(trainloader, 0):
                if data.size(0) != opt.batch_size:
                    continue

                lapnet2_gen.train(), lapnet2_disc.train()
                data_array = data.numpy()
                for i in range(opt.batch_size):
                    g4_target_temp = data_array[i]  # 3x64x64
                    g3_target_temp = g4_target_temp[:, ::2, ::2]  # 3x32x32
                    g2_target_temp = g3_target_temp[:, ::2, ::2]  # 3x16x16
                    g1_target_temp = g2_target_temp[:, ::2, ::2]  # 3x8x8

                    g2_target[i] = torch.from_numpy(g2_target_temp)
                    g1_target[i] = torch.from_numpy(g1_target_temp)

                    for j in range(channels):
                        g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))
                        y2[i, j, :] = torch.from_numpy(sensing_matrix2[j, :, :].dot(data_array[i, j].flatten()))

                g1_input_var = Variable(g1_input)
                # g2_input = lapnet1_gen(g1_input_var.detach())
                g2_input = lapnet1_gen(g1_input_var)
                # Train disc2 with true images
                lapnet2_disc.zero_grad()
                g2_target_var = Variable(g2_target)
                d2_output = lapnet2_disc(g2_target_var)
                d2_label_var = Variable(label.fill_(real_label))
                errD_d2_real_bce = criterion_bce(d2_output, d2_label_var)
                errD_d2_real_bce.backward()
                d2_real_mean = d2_output.data.mean()

                # Train disc2 with fake images
                y2_var = Variable(y2)
                g2_output = lapnet2_gen(g2_input, y2_var)
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

                # optimizer_lapnet1_gen.step()
                optimizer_lapnet2_gen.step()
                d2_fake_mean = d2_output.data.mean()

                if idx % opt.log_interval == 0:
                    g1_target_var = Variable(g1_target)
                    errD_g1_fake_mse = criterion_mse(g2_input, g1_target_var)

                    print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                          'errG1_mse: %.4f, D(x): %.4f, D(G(z)): %.4f' % (
                              2, epoch, epochs, idx, len(trainloader),
                              errD_d2_real_bce.data[0],
                              errD_d2_fake_bce.data[0],
                              errD_g2_fake_bce.data[0],
                              errD_g2_fake_mse.data[0],
                              errD_g1_fake_mse.data[0],
                              d2_real_mean,
                              d2_fake_mean))

            torch.save(lapnet2_gen.state_dict(), '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 2, epoch))
            torch.save(lapnet2_disc.state_dict(), '%s/%s/cr%s/%s/stage%s/model/disc_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 2, epoch))

            vutils.save_image(g2_target_var.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 2, epoch), normalize=True)
            vutils.save_image(g2_output.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 2, epoch), normalize=True)

            lapnet2_gen.eval(), lapnet2_disc.eval()
            val(epoch, 2, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3,
                sensing_matrix4, g2_target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen,
                criterion_mse, y2, y3, y4)

        # training level 3
        if opt.stage == 3 or opt.stage == 5:
            '''
            if epoch == 0 and opt.transfer == True:
              print('transfer weidhts from level1')
              lapnet3_gen.conv1 = copy.deepcopy(lapnet2_gen.conv1)
              lapnet3_gen.conv2 = copy.deepcopy(lapnet2_gen.conv2)
              lapnet3_gen.upsamp1 = copy.deepcopy(lapnet2_gen.upsamp1)
              lapnet3_gen.upsamp2 = copy.deepcopy(lapnet2_gen.upsamp2)
              lapnet3_gen.resblk1 = copy.deepcopy(lapnet2_gen.resblk1)

              lapnet3_disc.main[0] = copy.deepcopy(lapnet2_disc.main[0])
              lapnet3_disc.main[2] = copy.deepcopy(lapnet2_disc.main[2])
              lapnet3_disc.main[4] = copy.deepcopy(lapnet2_disc.main[4])
              lapnet3_disc.main[6] = copy.deepcopy(lapnet2_disc.main[5])
              lapnet3_disc.main[8] = copy.deepcopy(lapnet2_disc.main[8])
            '''

            for idx, (data, _) in enumerate(trainloader, 0):
                if data.size(0) != opt.batch_size:
                    continue

                lapnet3_gen.train(), lapnet3_disc.train()
                data_array = data.numpy()
                for i in range(opt.batch_size):
                    g4_target_temp = data_array[i]  # 1x64x64
                    g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                    g2_target_temp = g3_target_temp[:, ::2, ::2]  # 3x16x16
                    g1_target_temp = g2_target_temp[:, ::2, ::2]  # 3x8x8

                    g3_target[i] = torch.from_numpy(g3_target_temp)
                    g2_target[i] = torch.from_numpy(g2_target_temp)
                    g1_target[i] = torch.from_numpy(g1_target_temp)

                    for j in range(channels):
                        g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))
                        y2[i, j, :] = torch.from_numpy(sensing_matrix2[j, :, :].dot(data_array[i, j].flatten()))
                        y3[i, j, :] = torch.from_numpy(sensing_matrix3[j, :, :].dot(data_array[i, j].flatten()))

                g1_input_var = Variable(g1_input)
                g2_input = lapnet1_gen(g1_input_var)  # 1x8x8
                y2_var = Variable(y2)
                g3_input = lapnet2_gen(g2_input, y2_var)  # 1x16x16

                # Train disc3 with true images
                lapnet3_disc.zero_grad()
                g3_target_var = Variable(g3_target)
                d3_output = lapnet3_disc(g3_target_var)
                d3_label_var = Variable(label.fill_(real_label))
                errD_d3_real_bce = criterion_bce(d3_output, d3_label_var)
                errD_d3_real_bce.backward()
                d3_real_mean = d3_output.data.mean()
                # Train disc3 with fake images
                y3_var = Variable(y3)
                g3_output = lapnet3_gen(g3_input, y3_var)
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

                if idx % opt.log_interval == 0:
                    g1_target_var = Variable(g1_target)
                    g2_target_var = Variable(g2_target)
                    errD_g1_fake_mse = criterion_mse(g2_input, g1_target_var)
                    errD_g2_fake_mse = criterion_mse(g3_input, g2_target_var)

                    print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                          'errG2_mse: %.4f, errG1_mse: %.4f, D(x): %.4f, D(G(z)): %.4f' % (
                              3, epoch, epochs, idx, len(trainloader),
                              errD_d3_real_bce.data[0],
                              errD_d3_fake_bce.data[0],
                              errD_g3_fake_bce.data[0],
                              errD_g3_fake_mse.data[0],
                              errD_g2_fake_mse.data[0],
                              errD_g1_fake_mse.data[0],
                              d3_real_mean,
                              d3_fake_mean))

            torch.save(lapnet3_gen.state_dict(), '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 3, epoch))
            torch.save(lapnet3_disc.state_dict(), '%s/%s/cr%s/%s/stage%s/model/disc_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 3, epoch))

            vutils.save_image(g3_target_var.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 3, epoch), normalize=True)
            vutils.save_image(g3_output.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 3, epoch), normalize=True)

            lapnet3_gen.eval(), lapnet3_disc.eval()
            val(epoch, 3, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3,
                sensing_matrix4, g3_target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen,
                criterion_mse, y2, y3, y4)

        # training level 4
        if opt.stage == 4 or opt.stage == 5:
            for idx, (data, _) in enumerate(trainloader, 0):
                if data.size(0) != opt.batch_size:
                    continue

                lapnet4_gen.train(), lapnet4_disc.train()
                data_array = data.numpy()
                for i in range(opt.batch_size):
                    g4_target_temp = data_array[i]  # 1x64x64
                    g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                    g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
                    g1_target_temp = g2_target_temp[:, ::2, ::2]  # 1x8x8

                    g4_target[i] = torch.from_numpy(g4_target_temp)
                    g3_target[i] = torch.from_numpy(g3_target_temp)
                    g2_target[i] = torch.from_numpy(g2_target_temp)
                    g1_target[i] = torch.from_numpy(g1_target_temp)

                    for j in range(channels):
                        g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))
                        y2[i, j, :] = torch.from_numpy(sensing_matrix2[j, :, :].dot(data_array[i, j].flatten()))
                        y3[i, j, :] = torch.from_numpy(sensing_matrix3[j, :, :].dot(data_array[i, j].flatten()))
                        y4[i, j, :] = torch.from_numpy(sensing_matrix4[j, :, :].dot(data_array[i, j].flatten()))

                g1_input_var = Variable(g1_input)
                g2_input = lapnet1_gen(g1_input_var)  # 1x8x8
                y2_var = Variable(y2)
                g3_input = lapnet2_gen(g2_input, y2_var)  # 1x16x16
                y3_var = Variable(y3)
                g4_input = lapnet3_gen(g3_input, y3_var)  # 1x32x32

                # Train disc4 with true images
                g4_target_var = Variable(g4_target)
                lapnet4_disc.zero_grad()
                d4_output = lapnet4_disc(g4_target_var)
                d4_label_var = Variable(label.fill_(real_label))
                errD_d4_real_bce = criterion_bce(d4_output, d4_label_var)
                errD_d4_real_bce.backward()
                d4_real_mean = d4_output.data.mean()
                # Train disc4 with fake images
                y4_var = Variable(y4)
                g4_output = lapnet4_gen(g4_input, y4_var)
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

                if idx % opt.log_interval == 0:
                    g1_target_var = Variable(g1_target)
                    g2_target_var = Variable(g2_target)
                    g3_target_var = Variable(g3_target)

                    errD_g1_fake_mse = criterion_mse(g2_input, g1_target_var)
                    errD_g2_fake_mse = criterion_mse(g3_input, g2_target_var)
                    errD_g3_fake_mse = criterion_mse(g4_input, g3_target_var)

                    print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                          'errG3_mse: %.4f, errG2_mse: %.4f, errG1_mse: %.4f, D(x): %.4f, D(G(z)): %.4f' % (
                              4, epoch, epochs, idx, len(trainloader),
                              errD_d4_real_bce.data[0],
                              errD_d4_fake_bce.data[0],
                              errD_g4_fake_bce.data[0],
                              errD_g4_fake_mse.data[0],
                              errD_g3_fake_mse.data[0],
                              errD_g2_fake_mse.data[0],
                              errD_g1_fake_mse.data[0],
                              d4_real_mean,
                              d4_fake_mean))

            torch.save(lapnet4_gen.state_dict(), '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 4, epoch))
            torch.save(lapnet4_disc.state_dict(), '%s/%s/cr%s/%s/stage%s/model/disc_epoch_%d.pth'
                       % (opt.outf, opt.dataset, opt.cr, opt.model, 4, epoch))

            vutils.save_image(g4_target_var.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 4, epoch), normalize=True)
            vutils.save_image(g4_output.data,
                              '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                              % (opt.outf, opt.dataset, opt.cr, opt.model, 4, epoch), normalize=True)

            lapnet4_gen.eval(), lapnet4_disc.eval()
            val(epoch, 4, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3,
                sensing_matrix4, g4_target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen,
                criterion_mse, y2, y3, y4)

        # training the whole model from all the sub-models
        '''
        for idx, (data, _) in enumerate(trainloader, 0):
            if data.size(0) != opt.batch_size:
                continue

            data_array = data.numpy()
            for i in range(opt.batch_size):
                g4_target_temp = data_array[i]  # 1x64x64
                g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
                g1_target_temp = g2_target_temp[:, ::2, ::2]  # 1x8x8

                g4_target[i] = torch.from_numpy(g4_target_temp)
                g3_target[i] = torch.from_numpy(g3_target_temp)
                g2_target[i] = torch.from_numpy(g2_target_temp)
                g1_target[i] = torch.from_numpy(g1_target_temp)

                for j in range(channels):
                    g1_input[i, j, :] = torch.from_numpy(sensing_matrix1.dot(data_array[i, j].flatten()))
                    y2[i, j, :] = torch.from_numpy(sensing_matrix2.dot(data_array[i, j].flatten()))
                    y3[i, j, :] = torch.from_numpy(sensing_matrix3.dot(data_array[i, j].flatten()))
                    y4[i, j, :] = torch.from_numpy(sensing_matrix4.dot(data_array[i, j].flatten()))

            # Train lapnet_disc with true images
            lapnet1_disc.zero_grad(), lapnet2_disc.zero_grad()
            lapnet3_disc.zero_grad(), lapnet4_disc.zero_grad()

            g1_target_var, g2_target_var, g3_target_var, g4_target_var = Variable(g1_target), \
                                                                         Variable(g2_target), \
                                                                         Variable(g3_target), \
                                                                         Variable(g4_target)

            d1_output, d2_output, d3_output, d4_output = lapnet1_disc(g1_target_var),\
                                                         lapnet2_disc(g2_target_var),\
                                                         lapnet3_disc(g3_target_var),\
                                                         lapnet4_disc(g4_target_var)

            d1_label_var, d2_label_var, d3_label_var, d4_label_var = Variable(label.fill_(real_label)), \
                                                                     Variable(label.fill_(real_label)), \
                                                                     Variable(label.fill_(real_label)), \
                                                                     Variable(label.fill_(real_label))

            errD_d1_real_bce, errD_d2_real_bce, errD_d3_real_bce, errD_d4_real_bce = criterion_bce(d1_output, d1_label_var), \
                                                                                     criterion_bce(d2_output, d2_label_var), \
                                                                                     criterion_bce(d3_output, d3_label_var), \
                                                                                     criterion_bce(d4_output, d4_label_var)
            errD_d4_real_bce.backward()
            errD_d3_real_bce.backward()
            errD_d2_real_bce.backward()
            errD_d1_real_bce.backward()

            d1_real_mean, d2_real_mean, d3_real_mean, d4_real_mean = d1_output.data.mean(), \
                                                                     d2_output.data.mean(), \
                                                                     d4_output.data.mean(), \
                                                                     d4_output.data.mean()

            # Train lapnet_disc with fake images
            g1_input_var = Variable(g1_input)
            g2_input = lapnet1_gen(g1_input_var)  # 1x8x8
            y2_var = Variable(y2)
            g3_input = lapnet2_gen(g2_input, y2_var)  # 1x16x16
            y3_var = Variable(y3)
            g4_input = lapnet3_gen(g3_input, y3_var)  # 1x32x32
            y4_var = Variable(y4)
            g4_output = lapnet4_gen(g4_input, y4_var)

            # do not update gen1, gen2, gen3, gen4
            d1_output, d2_output, d3_output, d4_output = lapnet1_disc(g2_input.detach()), \
                                                         lapnet2_disc(g3_input.detach()), \
                                                         lapnet3_disc(g4_input.detach()), \
                                                         lapnet4_disc(g4_output.detach())

            d1_label_var, d2_label_var, d3_label_var, d4_label_var = Variable(label.fill_(fake_label)), \
                                                                     Variable(label.fill_(fake_label)), \
                                                                     Variable(label.fill_(fake_label)), \
                                                                     Variable(label.fill_(fake_label)),

            errD_d1_fake_bce, errD_d2_fake_bce,errD_d3_fake_bce, errD_d4_fake_bce = criterion_bce(d1_output, d1_label_var), \
                                                                                    criterion_bce(d2_output, d2_label_var), \
                                                                                    criterion_bce(d3_output, d3_label_var), \
                                                                                    criterion_bce(d4_output, d4_label_var)
            errD_d4_fake_bce.backward(), optimizer_lapnet4_disc.step()
            errD_d3_fake_bce.backward(), optimizer_lapnet3_disc.step()
            errD_d2_fake_bce.backward(), optimizer_lapnet2_disc.step()
            errD_d1_fake_bce.backward(), optimizer_lapnet1_disc.step()

            # Train lapnet_gen with fake images, lapgen_disc is not updated
            lapnet1_gen.zero_grad(), lapnet2_gen.zero_grad()
            lapnet3_gen.zero_grad(), lapnet4_gen.zero_grad()

            d1_label_var, d2_label_var, d3_label_var, d4_label_var = Variable(label.fill_(real_label)), \
                                                                     Variable(label.fill_(real_label)), \
                                                                     Variable(label.fill_(real_label)), \
                                                                     Variable(label.fill_(real_label))

            d1_output, d2_output, d3_output, d4_output = lapnet1_disc(g2_input), \
                                                         lapnet2_disc(g3_input), \
                                                         lapnet3_disc(g4_input), \
                                                         lapnet4_disc(g4_output)

            errD_g1_fake_bce, errD_g2_fake_bce, errD_g3_fake_bce, errD_g4_fake_bce = criterion_bce(d1_output, d1_label_var), \
                                                                                     criterion_bce(d2_output, d2_label_var), \
                                                                                     criterion_bce(d3_output, d3_label_var), \
                                                                                     criterion_bce(d4_output, d4_label_var)

            errD_g1_fake_mse, errD_g2_fake_mse, errD_g3_fake_mse, errD_g4_fake_mse = criterion_mse(g2_input, g1_target_var), \
                                                                                     criterion_mse(g3_input, g2_target_var), \
                                                                                     criterion_mse(g4_input, g3_target_var), \
                                                                                     criterion_mse(g4_output, g4_target_var)

            errD_g4 = opt.w_loss * errD_g4_fake_bce + (1 - opt.w_loss) * errD_g4_fake_mse

            errD_g4.backward()

            optimizer_lapnet4_gen.step()
            optimizer_lapnet3_gen.step()
            optimizer_lapnet2_gen.step()
            optimizer_lapnet1_gen.step()

            d1_fake_mean, d2_fake_mean, d3_fake_mean, d4_fake_mean = d1_output.data.mean(), \
                                                                     d2_output.data.mean(), \
                                                                     d3_output.data.mean(), \
                                                                     d4_output.data.mean()

            if idx % opt.log_interval == 0:
                print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                      'D(x): %.4f, D(G(z)): %.4f' % (
                          5, epoch, epochs, idx, len(trainloader),
                          errD_d4_real_bce.data[0],
                          errD_d4_fake_bce.data[0],
                          errD_g4_fake_bce.data[0],
                          errD_g4_fake_mse.data[0],
                          d4_real_mean,
                          d4_fake_mean))
                print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                      'D(x): %.4f, D(G(z)): %.4f' % (
                          3, epoch, epochs, idx, len(trainloader),
                          errD_d3_real_bce.data[0],
                          errD_d3_fake_bce.data[0],
                          errD_g3_fake_bce.data[0],
                          errD_g3_fake_mse.data[0],
                          d3_real_mean,
                          d3_fake_mean))
                print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                      'D(x): %.4f, D(G(z)): %.4f' % (
                          2, epoch, epochs, idx, len(trainloader),
                          errD_d2_real_bce.data[0],
                          errD_d2_fake_bce.data[0],
                          errD_g2_fake_bce.data[0],
                          errD_g2_fake_mse.data[0],
                          d2_real_mean,
                          d2_fake_mean))
                print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                      'D(x): %.4f, D(G(z)): %.4f \n' % (
                          1, epoch, epochs, idx, len(trainloader),
                          errD_d1_real_bce.data[0],
                          errD_d1_fake_bce.data[0],
                          errD_g1_fake_bce.data[0],
                          errD_g1_fake_mse.data[0],
                          d1_real_mean,
                          d1_fake_mean))

        val(epoch, 5, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3,
            sensing_matrix4, g4_target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen,
            criterion_mse, y2, y3, y4)

        torch.save(lapnet4_gen.state_dict(),
                   '%s/%s/%s/model/lapnet4_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet4_disc.state_dict(),
                   '%s/%s/%s/model/lapnet4_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet3_gen.state_dict(),
                   '%s/%s/%s/model/lapnet3_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet3_disc.state_dict(),
                   '%s/%s/%s/model/lapnet3_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet2_gen.state_dict(),
                   '%s/%s/%s/model/lapnet2_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet2_disc.state_dict(),
                   '%s/%s/%s/model/lapnet2_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet1_gen.state_dict(),
                   '%s/%s/%s/model/lapnet1_gen_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))
        torch.save(lapnet1_disc.state_dict(),
                   '%s/%s/%s/model/lapnet1_disc_epoch_%d.pth' % (opt.outf, opt.dataset, opt.model, epoch))

        vutils.save_image(g4_target_var.data,
                          '%s/%s/%s/image/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 5, epoch),
                          normalize=True)
        vutils.save_image(g4_output.data,
                          '%s/%s/%s/image/l%d_fake_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 5, epoch),
                          normalize=True)
        '''


def main():
    train_loader, val_loader = data_loader()
    train(opt.epochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
