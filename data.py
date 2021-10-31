import numpy as np
import os
import csv


def training():
	fieldnames = ['filename', 'label']
	rows = [[]]

	types = ('notumor', 'pituitary', 'meningioma', 'glioma')

	for i in range(len(types)):
		for filename in sorted(os.listdir(os.path.join("/home/akshay/Downloads/archive/Training/", types[i]))):
			row = []
			row.append(os.path.join(types[i], filename))
			row.append(i)
			rows.append(row)


	with open("/home/akshay/ml_project/labelled.csv", 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(fieldnames)
		writer.writerows(rows)

def test():
	fieldnames = ['filename', 'label']
	rows = [[]]

	types = ('notumor', 'pituitary', 'meningioma', 'glioma')

	for i in range(len(types)):
		for filename in sorted(os.listdir(os.path.join("/home/akshay/Downloads/archive/Testing/", types[i]))):
			row = []
			row.append(os.path.join(types[i], filename))
			row.append(i)
			rows.append(row)


	with open("/home/akshay/ml_project/labelled_test.csv", 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(fieldnames)
		writer.writerows(rows)

if __name__ == '__main__':
	test()