from flask import Flask, request
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn
import pickle

app = Flask(__name__)
pickle_in = open("randomforest.pkl",'rb')
random_forest = pickle.load(pickle_in)

@app.route('/')
def index():
	return "Hello, World"

if __name__ == '__main__':
	app.run(debug=True)