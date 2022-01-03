import torch
import torch.nn as nn 
import numpy as np 
import torchvision


class Network(nn.Module):
	def __init__(self, in_features, out_features):
		super(Network, self).__init__()
		self.model = nn.Sequential(
						nn.Linear(in_features, 256),
						nn.ReLU(),
						nn.Linear(256, out_features))

	def forward(self, x):
		return self.model(x)

class CNNNetwork(nn.Module):
	def __init__(self, out_features=4):
		super(CNNNetwork, self).__init__()
		self.model = torchvision.models.densenet169(pretrained=True)
		self.final_fcn = nn.Linear(1000, 4)

		for i, param in enumerate(self.model.features.parameters()):
			print(i)
			if(i < 400):
				param.requires_grad = False

	def forward(self, x):
		return self.final_fcn(self.model(x))

if __name__ == '__main__':
	model = CNNNetwork()
	x = torch.randn((5,3,224,224))
	i = 0
	for v in model.features.parameters():
		print(i, v)
		i += 1
	y = model(x)
	print(sum(p.numel() for p in model.parameters())/1000000	)

	print(y.shape)

