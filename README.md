"# LAPRAN-PyTorch"

This repository is an PyTorch implementation of the paper
**"LAPRAN: A Scalable Laplacian Pyramid Reconstructive Adversarial Network for Flexible Compressive Sensing Reconstruction"**.

## Code
Clone this repository into any place you want.
```bash
git clone
https://github.com/PSCLab-ASU/LAPRAN-PyTorch
cd LAPRAN-PyTorch

## Train your own model
You can start to train your own model via the following commands:

python main_adaptiveCS.py --model adaptiveCS_resnet_wy_ifusion_ufirst --dataset cifar10 --stage 1 --cr 20 --gpu 0

python main_adaptiveCS.py --model adaptiveCS_resnet_wy_ifusion_ufirst --dataset cifar10 --stage 2 --cr 20 --gpu 0

python main_adaptiveCS.py --model adaptiveCS_resnet_wy_ifusion_ufirst --dataset cifar10 --stage 3 --cr 20 --gpu 0

python main_adaptiveCS.py --model adaptiveCS_resnet_wy_ifusion_ufirst --dataset cifar10 --stage 4 --cr 20 --gpu 0

Then you get a four-stage LAPRAN for the cifar10 dataset.

## More codes of LAPRAN will be added to this repository later!

## Testing
The pretrained CIFAR10 model can be downloaded from: https://www.dropbox.com/s/eq2v2rowxqazj3u/results.zip?dl=0
Please download and upzip all files and directories to the root directory of the LAPRAN project.
. 
You can evaluate the pretrained model by running: python eval_adaptiveCS.py
