import torch 
from model import Network 
from dataset import MyDataset 
from torch.utils.data import DataLoader
from main import accuracy


if __name__ == '__main__':
	model = Network(588, 4)
	model.load_state_dict(torch.load("/home/akshay/ml_project/weights.pth"))
	model.eval()

	filename = "/home/akshay/ml_project/features_labelled_test.csv"
	dataset = MyDataset(filename, "testing")
	batch_data = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

	total_accuracy = 0.0
	data_no = 0.0

	for batch in batch_data:
		input_data, labels = batch['input'], batch['labels']
		predictions = model(input_data)
		data_no += input_data.shape[0]

		labels = torch.squeeze(labels, axis=1)
		total_accuracy += accuracy(predictions, labels)

	print("Test Accuracy: {:.4f}".format(total_accuracy / data_no))
