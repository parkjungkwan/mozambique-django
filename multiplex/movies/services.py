from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class DcGan(object):
    def __init__(self):
        # Root directory for dataset
        self.dataroot = r'C:\Users\AIA\PycharmProjects\djangoProject\multiplex\data'
        # Number of workers for dataloader
        self.workers = 2
        # Batch size during training
        self.batch_size = 128
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = 64
        # Number of channels in the training images. For color images this is 3
        self.nc = 3
        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100
        # Size of feature maps in generator
        self.ngf = 64
        # Size of feature maps in discriminator
        self.ndf = 64
        # Number of training epochs
        self.num_epochs = 10
        # Learning rate for optimizers
        self.lr = 0.0002
        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = 1
        self.manualSeed = 999

    def show_face(self):
        manualSeed = self.manualSeed
        dataroot = self.dataroot
        image_size = self.image_size
        batch_size = self.batch_size
        workers = self.workers
        ngpu = self.ngpu

        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers)
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def printNetG(self):
        # Create the generator
        ngpu = self.ngpu
        device = None
        netG = Generator(ngpu).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(self.weights_init)

        # Print the model
        print(netG)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        that = DcGan()
        nz = that.nz
        ngf = that.ngf
        nc = that.nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def spec(param):
    (lambda x: print(f"--- 1.Shape ---\n{x.shape}\n"
                     f"--- 2.Features ---\n{x.columns}\n"
                     f"--- 3.Info ---\n{x.info}\n"
                     f"--- 4.Case Top1 ---\n{x.head(1)}\n"
                     f"--- 5.Case Bottom1 ---\n{x.tail(3)}\n"
                     f"--- 6.Describe ---\n{x.describe()}\n"
                     f"--- 7.Describe All ---\n{x.describe(include='all')}"))(param)
dc_menu = ["Exit", #0
                "이미지 데이터셋 로딩",#1
                "NetG 출력",#2.
                "",#3
                "",#4
                "",#5
                "",#6
                ]
dc_lambda = {
    "1" : lambda x: x.show_face(),
    "2" : lambda x: x.printNetG(),
    "3" : lambda x: x.save_cctv_pop(),
    "4" : lambda x: x.save_police_norm(),
    "5" : lambda x: x.save_us_unemployment_map(),
    "6" : lambda x: x.save_seoul_crime_map(),
}
if __name__ == '__main__':
    dc = DcGan()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(dc_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                dc_lambda[menu](dc)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")