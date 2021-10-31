import torch
import torch.nn as nn 
import numpy as np 


class Network(nn.Module):
	def __init__(self, in_features, out_features):
		super(Network, self).__init__()
		self.model = nn.Sequential(
						nn.Linear(in_features, 256),
						nn.ReLU(),
						nn.Linear(256, out_features))

	def forward(self, x):
		return self.model(x)


if __name__ == '__main__':
	pass

