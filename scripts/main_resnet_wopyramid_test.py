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

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy.random import randn

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', help='basic | woresnet | woresnetup', default='resnetwopyramid')
parser.add_argument('--dataset', help='lsun | imagenet | mnist', default='folder')
parser.add_argument('--datapath', help='path to dataset', default='/home/user/kaixu/myGitHub/datasets/LSUN/')
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--layers-gan', type=int, default=3, metavar='N',
                    help='number of hierarchies in the GAN (default: 64)')
parser.add_argument('--gpu', type=int, default=2, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--w-loss', type=float, default=0.001, metavar='N.',
                    help='penalty for the mse and bce loss')
parser.add_argument('--lapnet-gen', default='./results/folder/resnetwopyramid/lapnet_gen_epoch_87.pth', help="path to gen (to continue training)")
parser.add_argument('--lapnet-disc', default='./results/folder/resnetwopyramid/lapnet_disc_epoch_87.pth', help="path to disc (to continue training)")

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

def test():
    # Initialize variables
    test_dataset = datasets.ImageFolder(root='/home/user/kaixu/myGitHub/datasets/BSDS500/test/',
                                        transform=transforms.Compose([
                                            transforms.Scale(opt.image_size),
                                            transforms.CenterCrop(opt.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))

    kwopt = {'num_workers': 2, 'pin_memory': True} if opt.cuda else {}
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    input, _ = testloader.__iter__().__next__()
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

    fake_label = 0
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

    lapnet_gen.load_state_dict(torch.load(opt.lapnet_gen))
    lapnet4_disc.load_state_dict(torch.load(opt.lapnet_disc))

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



    for idx, (data, _) in enumerate(testloader, 0):
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
        g4_target_var = Variable(g4_target)
        g4_output = lapnet_gen(g1_input_var, g1_input_var)
        errD_g4_fake_mse = criterion_mse(g4_output, g4_target_var)

        print('Test: [%d/%d] errG_mse: %.4f,' % (
                  idx, len(testloader),
                  errD_g4_fake_mse.data[0]))
        vutils.save_image(g4_target,
                          'test_%s/%s/%s/l%d_real_samples_epoch_%03d.png' % (opt.outf, opt.dataset, opt.model, 5, epoch),
                          normalize=True)
        vutils.save_image(g4_output.data,
                          'test_%s/%s/%s/l%d_fake_samples_epoch_%02d.png' % (opt.outf, opt.dataset, opt.model, 5, epoch),
                          normalize=True)

def main():
    test()

if __name__ == '__main__':
    main()
