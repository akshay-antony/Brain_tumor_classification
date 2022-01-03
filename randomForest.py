import sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

def train():
	data = pd.read_csv("/home/akshay/ml_project/features_labelled.csv")
	labels = data['label']
	input_data = np.load("total_features.npy")
	input_data = input_data / input_data.max(axis=0)
	
	test_data = pd.read_csv("features_labelled_test.csv")
	test_data_input = np.load("total_features_test.npy")
	test_data_input = test_data_input / test_data_input.max(axis=0)
	test_labels = test_data['label']
	
	random_forest = RandomForestClassifier(n_estimators=10, random_state=0)
	random_forest.fit(input_data, labels)
	
	pickle_out = open("randomforest.pkl",'wb')
	pickle.dump(random_forest, pickle_out)
	pickle_out.close()

	print("Training Accuracy: {:2f} ".format(random_forest.score(input_data, labels)*100), "%")
	print("Test Accuracy: {:2f}".format(random_forest.score(test_data_input, test_labels)*100), "%")


def inference():
	pickle_in = open("randomforest.pkl", 'rb')
	random_forest = pickle.load(pickle_in)
	test_data = pd.read_csv("features_labelled_test.csv")
	test_data_input = np.load("total_features_test.npy")
	test_data_input = test_data_input / test_data_input.max(axis=0)
	test_labels = test_data['label']
	print(test_data_input[0].shape)
	print(random_forest.predict(test_data_input[900].reshape(-1,588)), test_labels[900])

if __name__ == '__main__':
	inference()