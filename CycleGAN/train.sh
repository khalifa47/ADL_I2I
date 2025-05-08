!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
%cd pytorch-CycleGAN-and-pix2pix
!pip install dominate visdom

!bash ./datasets/download_cyclegan_dataset.sh horse2zebra

!python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
