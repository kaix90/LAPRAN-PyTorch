from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2
import models.lapgan_mnist_woresnet as lapgan
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy.random import randn

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', help='lsun | imagenet | mnist', default='mnist')
parser.add_argument('--datapath', help='path to dataset')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
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
parser.add_argument('--gpu', type=int, default=1, metavar='N',
                    help='which GPU do you want to use (default: 1)')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: please run with GPU")
print(opt)

if opt.seed is None:
    opt.seed = np.random.randint(1, 10000)
print('Random seed: ', opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True

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
        train_dataset = datasets.LSUN(db_path=opt.datapath, classes=['bedroom_train'],
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, **kwopt)

    return train_loader, test_loader

def train(epochs, trainloader):
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
    g1_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1, m2)
    g2_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1*2, m2*2)
    g3_input = torch.FloatTensor(opt.batch_size, sz_input[1], m1*4, m2*4)

    g1_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1, m2)
    g2_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1*2, m2*2)
    g3_target = torch.FloatTensor(opt.batch_size, sz_input[1], m1*4, m2*4)

    label = torch.FloatTensor(opt.batch_size)

    fake_label = 0
    real_label = 1

    # Instantiate models
    '''
    lapnet1 = lapgan.LAPGAN_Generator_level1(channels)
    lapnet2_disc = lapgan.LAPGAN_Discriminator_level2(channels)
    lapnet2_gen = lapgan.LAPGAN_Generator_level2(channels, m1*m2*channels)
    lapnet3_disc = lapgan.LAPGAN_Discriminator_level3(channels)
    lapnet3_gen = lapgan.LAPGAN_Generator_level3(channels, m1*m2*channels)
    '''
    lapnet1 = lapgan.LAPGAN_Generator_level1(channels, opt.ngpu)
    lapnet2_disc = lapgan.LAPGAN_Discriminator_level2(channels, opt.ngpu)
    lapnet2_gen = lapgan.LAPGAN_Generator_level2(channels, opt.ngpu)
    lapnet3_disc = lapgan.LAPGAN_Discriminator_level3(channels, opt.ngpu)
    lapnet3_gen = lapgan.LAPGAN_Generator_level3(channels, opt.ngpu)

    # Weight initialization
    lapnet1.apply(weights_init)
    lapnet2_disc.apply(weights_init)
    lapnet2_gen.apply(weights_init)
    lapnet3_disc.apply(weights_init)
    lapnet3_gen.apply(weights_init)

    print(lapnet1)
    print(lapnet2_disc)
    print(lapnet2_gen)
    print(lapnet3_disc)
    print(lapnet3_gen)

    optimizer_lapnet1 = optim.Adam(lapnet1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet2_disc = optim.Adam(lapnet2_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet2_gen = optim.Adam(lapnet2_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet3_disc = optim.Adam(lapnet3_disc.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_lapnet3_gen = optim.Adam(lapnet3_gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

#    torch.cuda.set_device(gpus[0])
    if opt.gpu:
        lapnet1.cuda()
        lapnet2_gen.cuda()
        lapnet2_disc.cuda()
        lapnet3_disc.cuda()
        lapnet3_gen.cuda()

        criterion_mse.cuda()
        criterion_bce.cuda()

        g1_input, g2_input, g3_input = g1_input.cuda(), g2_input.cuda(), g3_input.cuda()
        g1_target, g2_target, g3_target = g1_target.cuda(), g2_target.cuda(), g3_target.cuda()
        label = label.cuda()

    for epoch in range(epochs):

        # training level 1
        for idx, (data, _) in enumerate(trainloader, 0):
            data_array = data.numpy()
            sensing_matrix_left = randn(m1, n1)
            sensing_matrix_right = randn(n2, m2)

            for i in range(opt.batch_size):
                g3_target_temp = cv2.pyrDown(np.transpose(data_array[i], (1, 2, 0)))  # 1x32x32
                g2_target_temp = cv2.pyrDown(g3_target_temp)  # 1x16x16
                g2_target[i] = torch.from_numpy(np.expand_dims(g2_target_temp, axis=0))
                g1_target[i] = torch.from_numpy(np.expand_dims(cv2.pyrDown(g2_target_temp), axis=0))  # 1x8x8
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        sensing_matrix_left.dot(data_array[i, j]).dot(sensing_matrix_right))

            g1_input_var = Variable(g1_input)
            target_var = Variable(g1_target)

            optimizer_lapnet1.zero_grad()
            outputs = lapnet1(g1_input_var)
            loss = criterion_mse(outputs, target_var)
            loss.backward()
            optimizer_lapnet1.step()
            print('Level %d [%d/%d][%d/%d] loss_G: %.4f' % (1, epoch, epochs, idx, len(trainloader), loss.data[0]))

        # training level 2
        for idx, (data, _) in enumerate(trainloader, 0):
            data_array = data.numpy()
            sensing_matrix_left = randn(m1, n1)
            sensing_matrix_right = randn(n2, m2)

            for i in range(opt.batch_size):
                g3_target_temp = cv2.pyrDown(np.transpose(data_array[i], (1, 2, 0))) # 1x32x32
                g2_target_temp = cv2.pyrDown(g3_target_temp) # 1x16x16
                g2_target[i] = torch.from_numpy(np.expand_dims(g2_target_temp, axis=0))
                g1_target[i] = torch.from_numpy(np.expand_dims(cv2.pyrDown(g2_target_temp), axis=0)) # 1x8x8
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        sensing_matrix_left.dot(data_array[i, j]).dot(sensing_matrix_right))

            g1_input_var = Variable(g1_input)
            g1_output = lapnet1(g1_input_var)
            g1_output_cpu = g1_output.cpu().data.numpy()
            for i in range(opt.batch_size):
                g2_input[i] = torch.from_numpy(np.expand_dims(cv2.pyrUp(np.transpose(g1_output_cpu[i], (1, 2, 0))), axis=0)) # 1x14x14

            # Train disc2 with true images
            g2_input_var = Variable(g2_input)
            g2_target_var = Variable(g2_target)
            optimizer_lapnet2_disc.zero_grad()
            d2_output = lapnet2_disc(g2_target_var - g2_input_var)
            d2_label_var = Variable(label.fill_(real_label))
            errD_d2_real_bce = criterion_bce(d2_output, d2_label_var)
            errD_d2_real_bce.backward()
            d2_real_mean = d2_output.data.mean()

            # Train disc2 with fake images
        #    g2_output = lapnet2_gen(g2_input_var, g1_input_var)
            g2_output = lapnet2_gen(g2_input_var)
            d2_output = lapnet2_disc(g2_output.detach())
            #loss_mse = criterion_mse(outputs, g2_target_var)
            d2_label_var = Variable(label.fill_(fake_label))
            errD_d2_fake_bce = criterion_bce(d2_output, d2_label_var)
            errD_d2_fake_bce.backward()
            optimizer_lapnet2_disc.step()

            # Train gen2 with fake images, disc2 is not updated
            optimizer_lapnet2_gen.zero_grad()
            d2_label_var = Variable(label.fill_(real_label))
            d2_output = lapnet2_disc(g2_output)
            errD_g2_fake_bce = criterion_bce(d2_output, d2_label_var)
            errD_g2_fake_mse = criterion_mse(g2_output, g2_target_var)
            errD_g2 = 0.2 * errD_g2_fake_bce + 0.8 * errD_g2_fake_mse
            errD_g2_fake_bce.backward()
            optimizer_lapnet2_gen.step()
            d2_fake_mean = d2_output.data.mean()

            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f' % (
                2, epoch, epochs, idx, len(trainloader),
                d2_real_mean,
                d2_fake_mean,
                errD_g2_fake_bce.data[0],
                errD_g2_fake_mse.data[0]))

        # training level 3
        for idx, (data, _) in enumerate(trainloader, 0):
            data_array = data.numpy()
            sensing_matrix_left = randn(m1, n1)
            sensing_matrix_right = randn(n2, m2)
            for i in range(opt.batch_size):
                g3_target_temp = cv2.pyrDown(np.transpose(data_array[i], (1, 2, 0)))  # 1x32x32
                g2_target_temp = cv2.pyrDown(g3_target_temp)  # 1x16x16
                g3_target[i] = torch.from_numpy(np.expand_dims(g3_target_temp, axis=0))
                g2_target[i] = torch.from_numpy(np.expand_dims(g2_target_temp, axis=0))
                g1_target[i] = torch.from_numpy(np.expand_dims(cv2.pyrDown(g2_target_temp), axis=0))  # 1x8x8
                for j in range(channels):
                    g1_input[i, j, :, :] = torch.from_numpy(
                        sensing_matrix_left.dot(data_array[i, j]).dot(sensing_matrix_right))
            g1_input_var = Variable(g1_input)
            g2_input_var = Variable(g2_input)
            g2_output = lapnet2_gen(g2_input_var)   # 1x16x16
            g2_output_cpu = g2_output.cpu().data.numpy()
            for i in range(opt.batch_size):
                g3_input[i] = torch.from_numpy(
                    np.expand_dims(cv2.pyrUp(np.transpose(g2_output_cpu[i], (1, 2, 0))), axis=0))  # 1x32x32
            # Train disc3 with true images
            g3_input_var = Variable(g3_input)
            g3_target_var = Variable(g3_target)
            optimizer_lapnet3_disc.zero_grad()
            d3_output = lapnet3_disc(g3_target_var)
            d3_label_var = Variable(label.fill_(real_label))
            errD_d3_real_bce = criterion_bce(d3_output, d3_label_var)
            errD_d3_real_bce.backward()
            # Train disc3 with fake images
            #g3_output = lapnet3_gen(g3_input_var, g1_input_var)
            g3_output = lapnet3_gen(g3_input_var)
            d3_output = lapnet3_disc(g3_output.detach())
            # loss_mse = criterion_mse(outputs, g2_target_var)
            d3_label_var = Variable(label.fill_(fake_label))
            errD_d3_fake_bce = criterion_bce(d3_output, d3_label_var)
            errD_d3_fake_bce.backward()
            optimizer_lapnet3_disc.step()
            # Train gen3 with fake images, disc3 is not updated
            optimizer_lapnet3_gen.zero_grad()
            d3_label_var = Variable(label.fill_(real_label))
            d3_output = lapnet3_disc(g3_output)
            errD_g3_fake_bce = criterion_bce(d3_output, d3_label_var)
            errD_g3_fake_bce.backward()
            optimizer_lapnet3_gen.step()
            print('Level %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f' % (
                2, epoch, epochs, idx, len(trainloader),
                errD_d2_real_bce.data[0],
                errD_d2_fake_bce.data[0],
                errD_g2_fake_bce.data[0],
                errD_g2_fake_mse.data[0]))

        torch.save(lapnet1.state_dict(), '%s/g1_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(lapnet2_disc.state_dict(), '%s/d2_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(lapnet2_gen.state_dict(), '%s/g2_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(lapnet3_disc.state_dict(), '%s/d3_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(lapnet3_gen.state_dict(), '%s/g3_epoch_%d.pth' % (opt.outf, epoch))

        vutils.save_image(g3_input, '%s/real_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)
        vutils.save_image(g3_output.data, '%s/fake_samples_epoch_%03d.png' %(opt.outf, epoch), normalize=True)

def main():
    train_loader, test_loader = data_loader()
    train(opt.epochs, train_loader)

if __name__ == '__main__':
    main()