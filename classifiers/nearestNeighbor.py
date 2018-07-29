import numpy as np

class nearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        # X is (N x D) where each row is an example.
        # y is 1 dimensional of size N.
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        numTest = X.shape[0]
        Ypred = np.zeros(numTest, dtype=self.ytr.dtype)

        for i in np.arange(numTest):
            distances = np.sum(abs(self.Xtr - X[i, :]), axis=1)
            minIndex = np.argmin(distances)
            Ypred[i] = self.ytr[minIndex]

        return Ypred
