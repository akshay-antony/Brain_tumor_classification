import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class MyDataset(Dataset):
	def __init__(self, filename, base_path):
		super(MyDataset, self).__init__()
		data = pd.read_csv(filename)
		filenames = data['filename']
		labels = data['label']
		self.x = filenames 
		self.y = labels
		self.base_path = "/home/akshay/ml_project/" + base_path 

	def __getitem__(self, idx):
		x_ = np.load(os.path.join(self.base_path, self.x[idx]))

		for i in range(0, 588, 196):
			x_[i:i+196] /= np.max(x_[i:i+196])

		x_ = torch.from_numpy(x_)
		y_ = torch.Tensor([self.y[idx]])
		
		x_ = x_.type(torch.float32)
		y_ = y_.type(torch.LongTensor)

		sample_data = {'input': x_, 'labels': y_}
		return sample_data

	def __len__(self):
		return len(self.x)


if __name__ == '__main__':
	pass
	# filename = "/home/akshay/ml_project/features_labelled.csv"
	# dataset = MyDataset(filename)

	# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)

	# for batch in dataloader:
	# 	x = batch['input']
	# 	y = batch['labels']
	# 	print(x.shape, y.shape)