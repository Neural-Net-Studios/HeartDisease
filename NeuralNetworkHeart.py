from NeuralNetStd import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('C:/Users/asynk/Desktop/Data/heart.csv')
x = data.drop(['target'], axis=1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
clf = NeuralNetwork()
# Input Layer
clf.inputlayer(200, 'sigmoid', 13)
# Hidden Layers
clf.hiddenlayer(0.3, 150, 'sigmoid')
# Output Layer
clf.outputlayer(1, 'sigmoid')
# Training
clf.training('adam', 'mse', ['accuracy'], x_train, y_train, 900)
# Test
clf.test(x_test, y_test)
# Predict
import numpy as np
clf.predict(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))