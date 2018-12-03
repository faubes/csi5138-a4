# CSI5138 Assignment 4
# Data Loader
# Joel Faubert
# 2560106
# 
# Functions to load MNIST / CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader():
	def __init__(self, batch_size=128, num_workers=0):
		self.batch_size=batch_size
		self.num_workers=num_workers

	
	def load_cifar10(self):
		transform = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			
		trainset = torchvision.datasets.CIFAR10(root='D:/datasets/cifar10', train=True, download=True, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

		testset = torchvision.datasets.CIFAR10(root='D:/datasets/cifar10', train=False, download=True, transform=transform)

		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		
		return trainloader, testloader, classes
		
	def load_mnist(self):
		transform = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			
		trainset = torchvision.datasets.MNIST(root='D:/datasets/mnist', train=True, download=True, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

		testset = torchvision.datasets.MNIST(root='D:/datasets/mnist', train=False, download=True, transform=transform)

		testloader = torch.utils.data.DataLoader(testset, batch_size=4,
			shuffle=False, num_workers=2)

		classes = range(10)
		
		return trainloader, testloader, classes
		