#!/usr/local/bin/python

import pandas as pd
import numpy as np

from Perceptron import Perceptron as Pr

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length- two features -> 3 weights
X = df.iloc[0:100, [0,2]].values

perc = Pr(epochs=10, eta=0.1)

perc.train(X, y)
print('Weights: %s' % perc.weights)
print('Errors: %s' % perc.errors)
