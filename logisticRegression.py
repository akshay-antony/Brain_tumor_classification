from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


if __name__ == '__main__':
	data = pd.read_csv("/home/akshay/ml_project/features_labelled.csv")
	labels = data['label']
	input_data = np.load("total_features.npy")
	
	test_data = pd.read_csv("features_labelled_test.csv")
	test_data_input = np.load("total_features_test.npy")
	test_labels = test_data['label']
	
	input_data = input_data / input_data.max(axis=0)
	test_data_input = test_data_input / test_data_input.max(axis=0)

	reg = LogisticRegression(max_iter=10000)
	reg.fit(input_data, labels)

	print("Training Accuracy: ", reg.score(input_data, labels))
	print("Test Accuracy: ", reg.score(test_data_input, test_labels))