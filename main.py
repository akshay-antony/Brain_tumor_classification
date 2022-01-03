import torch 
from model import Network
from dataset import MyDataset
import torch.nn as nn
from torch.utils.data import DataLoader

def accuracy(predictions, labels):
	_, predicted = torch.max(predictions, dim=1)
	accuracy = torch.sum(predicted == labels).item()
	return accuracy

if __name__ == '__main__':
	model = Network(588, 4)
	model.train()

	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

	filename = "/home/akshay/ml_project/features_labelled.csv"
	dataset = MyDataset(filename, "training")
	epochs = 100

	for j in range(epochs):
		batch_data = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

		epoch_loss = 0
		data_no = 0 
		total_acc = 0

		for i, batch_data in enumerate(batch_data):
			input_data = batch_data['input']
			labels = batch_data['labels']
			labels = torch.squeeze(labels, axis=1)
			print(labels)
			predictions = model(input_data)
			print(predictions)
			optimizer.zero_grad()

			loss = loss_func(predictions, labels)
			epoch_loss += loss.item() * input_data.shape[0]
			data_no += input_data.shape[0]
			total_acc += accuracy(predictions, labels)

			loss.backward()
			optimizer.step()

		print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}".format(j, epoch_loss/data_no, total_acc/data_no))

	path = "/home/akshay/ml_project/weights.pth"
	torch.save(model.state_dict(), path)