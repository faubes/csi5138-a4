# # CSI5138 Assignment 4
# WGAN
# Joel Faubert
# 2560106
# 
# Based on Pytorch DCGAN tutorial from 
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and 
# https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
# https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py


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
import load_data

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
		
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
		

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch VAE')
	parser.add_argument('--batch_size', type=int, default=128, metavar='N',
						help='input batch size for training (default: 128)')
	parser.add_argument('--n_critic', type=int, default=10, metavar='N',
						help='update generator every n_critic steps (default: 5)')
	parser.add_argument('--use_mnist', action='store_true', default=False, 
						help='Use MNIST dataset (default: True)')
	parser.add_argument('--use_cifar10', action='store_true', default=True, 
						help='Use CIFAR10 dataset (default: False)')
	parser.add_argument('--epochs', type=int, default=50, metavar='N',
						help='number of epochs to train (default: 20)')
	parser.add_argument('--num_workers', type=int, default=4, metavar='W',
						help='number of workers to use for data loader (default: 4)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=999, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')

	args = parser.parse_args()

	args.cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda else "cpu")

	#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	num_epochs = args.epochs
	n_critic = args.n_critic
	
	datapath = 'D:/datasets/'

	loader = load_data.DataLoader(datapath, args.batch_size, args.num_workers)

	if args.use_mnist:
		train_loader, test_loader, classes = loader.load_mnist()
		# Number of channels in the training images. For color images this is 3
		nc = 1
		filestr = "mnist_wgan_"


	if args.use_cifar10:
		train_loader, test_loader, classes = loader.load_cifar10()
		nc = 3
		filestr = "cifar10_wgan_"

		
	# Set random seem for reproducibility
	manualSeed = args.seed
	#manualSeed = random.randint(1, 10000) # use if you want new results
	print("Random Seed: ", manualSeed)

	random.seed(manualSeed)

	torch.manual_seed(manualSeed)

	# Root directory for dataset
	dataroot = "D:/datasets/"

	# Size of z latent vector (i.e. size of generator input)
	nz = 100

	# Size of feature maps in generator
	ngf = 64

	# Size of feature maps in discriminator
	ndf = 64

	# Learning rate for optimizers
	lr = 0.0002

	# Beta1 hyperparam for Adam optimizers
	beta1 = 0.5

	# Number of GPUs available. Use 0 for CPU mode.
	ngpu = 1

	# Decide which device we want to run on
	device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

	# Plot some training images
	real_batch = next(iter(train_loader))
	#plt.figure(figsize=(8,8))
	#plt.axis("off")
	#plt.title("Training Images")
	#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
	#plt.savefig('gan_test.png')

	# Create the generator
	netG = Generator(ngpu).to(device)

	# Handle multi-gpu if desired
	if (device.type == 'cuda') and (ngpu > 1):
		netG = nn.DataParallel(netG, list(range(ngpu)))

	# Apply the weights_init function to randomly initialize all weights
	#  to mean=0, stdev=0.2.
	netG.apply(weights_init)

	# Print the model
	print(netG)

	# Create the Discriminator
	netD = Discriminator(ngpu).to(device)

	# Handle multi-gpu if desired
	if (device.type == 'cuda') and (ngpu > 1):
		netD = nn.DataParallel(netD, list(range(ngpu)))

	# Apply the weights_init function to randomly initialize all weights
	#  to mean=0, stdev=0.2.
	netD.apply(weights_init)

	# Print the model
	print(netD)

	# Create batch of latent vectors that we will use to visualize
	#  the progression of the generator
	fixed_noise = torch.randn(64, nz, 1, 1, device=device)

	# Establish convention for real and fake labels during training
	real_label = 1
	fake_label = 0

	# Setup Adam optimizers for both G and D
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

	# Training Loop

	# Lists to keep track of progress
	img_list = []
	G_losses = []
	D_losses = []
	iters = 0


	print("Starting Training Loop...")
	
	# For each epoch
	for epoch in range(num_epochs):
		# For each batch in the dataloader
		for i, data in enumerate(train_loader, 0):
		
			# calculate step
			
			step = epoch * len(train_loader) + i + 1
			# Dicriminator forward-loss-backward-update
			# Format batch
			real_cpu = data[0].to(device)
			b_size = real_cpu.size(0)
			#label = torch.full((b_size,), real_label, device=device)
			
			# Generate batch of latent vectors
			noise = torch.randn(b_size, nz, 1, 1, device=device)
			
			# Generate fake image batch with G
			G_sample = netG(noise)
			
			# Forward pass real batch through D
			D_real = netD(real_cpu).view(-1)
			D_fake = netD(G_sample.detach())
					
			D_loss = -(torch.mean(D_real) - torch.mean(D_fake)) # Wasserstein-1 Distance		

			netD.zero_grad()
			
			# autograd
			D_loss.backward()
			
			# Update D
			optimizerD.step()
			
			 # Weight clipping
			for p in netD.parameters():
				p.data.clamp_(-0.01, 0.01)
			
			if i % n_critic == 0:
			
				############################
				# (2) Update G network:
				###########################
				netD.zero_grad()
				netG.zero_grad()
				# Since we just updated D, perform another forward pass of all-fake batch through D
				output = netD(G_sample).view(-1)
				# Calculate G's loss based on this output
				G_loss = -torch.mean(output)
				# Calculate gradients for G
				G_loss.backward()
				D_G_z2 = output.mean().item()
				# Update G
				optimizerG.step()

			#Print and plot every now and then
			if i % 1000 == 0:
				print('Epoch-{}; Step-{}; D_loss: {}; G_loss: {}'.format(epoch, step, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
				
			# Save Losses for plotting later
			G_losses.append(G_loss.item())
			D_losses.append(D_loss.item())

			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
				with torch.no_grad():
					fake = netG(fixed_noise).detach().cpu()
				img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

			iters += 1
	

	# plot D & G losses during training
	plt.figure(figsize=(10,5))
	plt.title("Generator and Critic Loss During Training")
	plt.plot(G_losses,label="G")
	plt.plot(D_losses,label="C")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(str(filestr + "_loss_plot.png"))
	plt.close()
	
	# Grab a batch of real images from the dataloader
	real_batch = next(iter(test_loader))

	# Plot the real images
	plt.figure(figsize=(15,15))
	plt.subplot(1,3,1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

	plt.subplot(1,3,2)
	plt.axis("off")
	plt.title("Fake Images after " + str(num_epochs//2) + " epochs")
	plt.imshow(np.transpose(img_list[num_epochs//2 +1],(1,2,0)))

	# Plot the fake images from the last epoch
	plt.subplot(1,3,3)
	plt.axis("off")
	plt.title("Fake Images after " + str(num_epochs) + " epochs")
	plt.imshow(np.transpose(img_list[-1],(1,2,0)))
	plt.savefig(str(filestr + "real_v_fake.png"))
	plt.close()
