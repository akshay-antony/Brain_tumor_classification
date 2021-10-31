import numpy as np
import os
import csv


if __name__ == '__main__':

	fieldnames = ['filename', 'label']
	rows = [[]]

	types = ('notumor', 'pituitary', 'meningioma', 'glioma')

	for i in range(len(types)):
		for filename in sorted(os.listdir(os.path.join("/home/akshay/ml_project/training/" + types[i]))):
			row = []
			row.append(os.path.join(types[i], filename))
			row.append(i)
			rows.append(row)


	with open("/home/akshay/ml_project/features_labelled.csv", 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(fieldnames)
		writer.writerows(rows)

