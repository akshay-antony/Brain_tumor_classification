import torch 
from model import Network
from data_prepare import calculate_features
import cv2
import numpy as np
import pandas as pd


patch_size = 10

locations = []
labels = ('notumor', 'pituitary', 'meningioma', 'glioma')

for i in range(28, 224, 28):
	for j in range(28, 224, 28):
		locations.append([i,j])

def normalize(x_):
	for i in range(0, 588, 196):
		x_[i:i+196] /= np.max(x_[i:i+196])

	return x_

if __name__ == '__main__':
	data = pd.read_csv('/home/akshay/ml_project/labelled_test.csv')
	row = data['filename']
	row_type = data['label']
	i = np.random.randint(0, 1313)
	filename = "/home/akshay/Downloads/archive/Testing/" + row[i]

	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(image, (224, 224))

	glcm_features = calculate_features(resized_image)
	
	model = Network(588, 4)
	model.load_state_dict(torch.load("/home/akshay/ml_project/weights.pth"))
	model.eval()

	input_data = normalize(glcm_features)
	input_data = torch.from_numpy(input_data)
	input_data = input_data.type(torch.FloatTensor)
	input_data = torch.unsqueeze(input_data, dim=0)

	prediction = model(input_data)

	output = torch.argmax(prediction)

	print("prediction:", labels[int(output)], "correct label:", labels[int(row_type[i])])
