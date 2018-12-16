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
import skimage.io as sio
import time

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | adaptiveCS | adaptiveCS_resnet',
                    default='adaptiveCS_resnet_wy_ifusion_ufirst')  # adaptiveCS_resnet_wy_lfusion_ufirst
parser.add_argument('--dataset', help='lsun | imagenet | mnist | bsd500 | bsd500_patch', default='cifar10')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/CSImageNet/data/')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
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
parser.add_argument('--stage', type=int, default=1, help='the stage under training')
parser.add_argument('--transfer', action='store_true', default=False, help='transfer weights')
parser.add_argument('--cr', type=int, default=30, help='compression ratio')
parser.add_argument('--Set', type=int, default=0, metavar='N',
                    help='Set 5 or Set 14')

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

criterion_mse = nn.MSELoss()

cudnn.benchmark = True

if not os.path.exists('%s/%s/cr%s/%s/test' % (opt.outf, opt.dataset, opt.cr, opt.model)):
    os.makedirs('%s/%s/cr%s/%s/test' % (opt.outf, opt.dataset, opt.cr, opt.model))


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
        if opt.Set == 5:
            test_dataset = datasets.ImageFolder(root=opt.datapath + 'test_set5_64x64',
                                                transform=transforms.Compose([
                                                    transforms.Resize(opt.image_size),
                                                    transforms.CenterCrop(opt.image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))
        elif opt.Set == 14:
            test_dataset = datasets.ImageFolder(root=opt.datapath + 'test_set14_64x64',
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


def evaluation(testloader):
    # Initialize variables
    input, _ = testloader.__iter__().__next__()
    input = input.numpy()
    sz_input = input.shape
    cr1 = 8 * opt.cr
    cr2 = 4 * opt.cr
    cr3 = 2 * opt.cr
    cr4 = opt.cr
    channels = sz_input[1]
    n = sz_input[2] ** 2
    m1 = n / cr1
    m2 = n / cr2
    m3 = n / cr3
    m4 = n / cr4
    img_size1 = sz_input[3] / 8
    img_size2 = sz_input[3] / 4
    img_size3 = sz_input[3] / 2
    img_size4 = sz_input[3]

    sensing_matrix4 = np.load('sensing_matrix_cr' + str(opt.cr) + '.npy')
    # sensing_matrix4 = randn(channels, m4, n)
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

    # Instantiate models
    lapnet1_gen = lapgan.LAPGAN_Generator_level1(channels, channels * m1, opt.ngpu)
    lapnet2_gen = lapgan.LAPGAN_Generator_level2(channels, channels * m2, opt.ngpu)
    lapnet3_gen = lapgan.LAPGAN_Generator_level3(channels, channels * m3, opt.ngpu)
    lapnet4_gen = lapgan.LAPGAN_Generator_level4(channels, channels * m4, opt.ngpu)

    if opt.cuda:
        lapnet1_gen.cuda()
        lapnet2_gen.cuda()
        lapnet3_gen.cuda()
        lapnet4_gen.cuda()

        criterion_mse.cuda()

        g1_input, g2_input, g3_input, g4_input = g1_input.cuda(), g2_input.cuda(), g3_input.cuda(), g4_input.cuda()
        g1_target, g2_target, g3_target, g4_target = g1_target.cuda(), g2_target.cuda(), g3_target.cuda(), g4_target.cuda()
        y2, y3, y4 = y2.cuda(), y3.cuda(), y4.cuda()

    if opt.dataset == 'bsd500_patch':
        if 'wy' in opt.model and 'ifusion' in opt.model:
            if opt.cr == 5:
                level1_iter = 12 # 0.0260
                level2_iter = 17 # 0.0153
                level3_iter = 10 # 0.0096 modify
                level4_iter = 11 # 0.0053 modify
            elif opt.cr == 10:
                level1_iter = 12  # 0.0350 18  # 14
                level2_iter = 12  # 0.0225 16  # 27
                level3_iter = 20  # 0.0148 8  # 27
                level4_iter = 7  # 0.0093
            elif opt.cr == 20:
                level1_iter = 6  # 0.0447 3
                level2_iter = 9  # 0.0322 9
                level3_iter = 16  # 0.0215
                level4_iter = 6  # 0.0144
            elif opt.cr == 30:
                level1_iter = 7  # 0.0550
                level2_iter = 7  # 0.0386
                level3_iter = 12  # 0.0283
                level4_iter = 7  # 0.0190
    if opt.dataset == 'mnist':
        if 'wy' in opt.model and 'ifusion' in opt.model:
            if opt.cr == 5:
                level1_iter = 80  # 0.0070 18  # 14
                level2_iter = 36  # 0.0059 16  # 27
                level3_iter = 97  # 0.0025 8  # 27
                level4_iter = 6   # 0.0022
            elif opt.cr == 10:
                level1_iter = 53  # 0.0206  18  # 14
                level2_iter = 43  # 0.0100  16  # 27
                level3_iter = 95  # 0.0113  8  # 27
                level4_iter = 10  # 0.0027
            elif opt.cr == 20:
                level1_iter = 38  # 0.0475  18  # 14
                level2_iter = 43  # 0.0219  16  # 27
                level3_iter = 23  # 0.0148  8  # 27
                level4_iter = 21  # 0.0043
            elif opt.cr == 30:
                level1_iter = 34  # 0.0740  18  # 14
                level2_iter = 55  # 0.0365  16  # 27
                level3_iter = 98  # 0.0242  8  # 27
                level4_iter = 28  # 0.0061

    if opt.dataset == 'cifar10':
        if 'wy' in opt.model and 'ifusion' in opt.model:
            if opt.cr == 5:
                level1_iter = 28 # 0.0126
                level2_iter = 81 # 0.0039
                level3_iter = 21 # 0.0025
                level4_iter = 26  # 0.0008
            elif opt.cr == 10:
                level1_iter = 24 # 0.0242
                level2_iter = 33 # 0.0102
                level3_iter = 58 # 0.0045
                level4_iter = 21 # 0.0017
            elif opt.cr == 20:
                level1_iter = 19 # 0.0420
                level2_iter = 30 # 0.0221
                level3_iter = 44 # 0.0111
                level4_iter = 21 # 0.0043
            elif opt.cr == 30:
                level1_iter = 16 # 0.0532
                level2_iter = 34 # 0.0317
                level3_iter = 84 # 0.0174
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
    if opt.stage == 5:
        stage1_path = '%s/%s/cr%s/%s/stage5/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, level1_iter)
        stage2_path = '%s/%s/cr%s/%s/stage5/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, level2_iter)
        stage3_path = '%s/%s/cr%s/%s/stage5/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, level3_iter)
        stage4_path = '%s/%s/cr%s/%s/stage5/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, level4_iter)
    else:
        stage1_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, 1, level1_iter)
        stage2_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, 2, level2_iter)
        stage3_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, 3, level3_iter)
        stage4_path = '%s/%s/cr%s/%s/stage%s/model/gen_epoch_%d.pth' % (
            opt.outf, opt.dataset, opt.cr, opt.model, 4, level4_iter)

    lapnet1_gen.load_state_dict(torch.load(stage1_path))
    lapnet2_gen.load_state_dict(torch.load(stage2_path))
    lapnet3_gen.load_state_dict(torch.load(stage3_path))
    lapnet4_gen.load_state_dict(torch.load(stage4_path))

    lapnet1_gen.eval()
    lapnet2_gen.eval()
    lapnet3_gen.eval()
    lapnet4_gen.eval()

    errD_fake_mse_total = 0
    elapsed_time = 0

    for idx, (data, _) in enumerate(testloader, 0):
        data_array = data.numpy()
        for i in range(opt.batch_size):
            g4_target_temp = data_array[i]  # 1x64x64
            g4_target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64

            for j in range(channels):
                g1_input[i, j, :] = torch.from_numpy(sensing_matrix1[j, :, :].dot(data_array[i, j].flatten()))
                y2[i, j, :] = torch.from_numpy(sensing_matrix2[j, :, :].dot(data_array[i, j].flatten()))
                y3[i, j, :] = torch.from_numpy(sensing_matrix3[j, :, :].dot(data_array[i, j].flatten()))
                y4[i, j, :] = torch.from_numpy(sensing_matrix4[j, :, :].dot(data_array[i, j].flatten()))

        g1_input_var = Variable(g1_input, volatile=True)
        y2_var = Variable(y2)
        y3_var = Variable(y3)
        y4_var = Variable(y4)

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.time()

        g2_input = lapnet1_gen(g1_input_var)
        g3_input = lapnet2_gen(g2_input, y2_var)
        g4_input = lapnet3_gen(g3_input, y3_var)
        g4_output = lapnet4_gen(g4_input, y4_var)

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
