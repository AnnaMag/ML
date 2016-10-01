import numpy as np


class Perceptron(object):

    def __init__(self, eta=0.01, epochs=1):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        #initialize the weights
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        # for a number of epochs
        for _ in range(self.epochs):
            self.errors_it = 0;
            [self.update(xi, target) for xi, target in zip(X, y)]
            self.errors.append(self.errors_it);
        return self

    def project(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0];

    def update(self, x, target):
        target_vs_output = self.eta * (target - self.predict(x));


        self.weights[1:] +=  target_vs_output * x;
        # x0 = 1, so
        self.weights[0] +=  target_vs_output;
        # number of misclassifications
        self.errors_it += int(target_vs_output != 0.0);
        return self
    def predict(self, X):
        return np.where(self.project(X) >= 0.0, 1, -1)
