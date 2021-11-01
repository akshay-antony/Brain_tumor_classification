from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


if __name__ == '__main__':
	data = pd.read_csv("/home/akshay/ml_project/features_labelled.csv")
	labels = data['label']
	input_data = np.load("total_features.npy")
	
	test_data = pd.read_csv("features_labelled_test.csv")
	test_data_input = np.load("total_features_test.npy")
	test_labels = test_data['label']
	
	input_data = input_data / input_data.max(axis=0)
	test_data_input = test_data_input / test_data_input.max(axis=0)

	svc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter=-1))
	svc_clf.fit(input_data, labels)
	
	print("Training Accuracy: {:2f} ".format(svc_clf.score(input_data, labels)*100), "%")
	print("Test Accuracy: {:2f}".format(svc_clf.score(test_data_input, test_labels)*100), "%")
