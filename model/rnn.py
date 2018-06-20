import numpy as np
from sklearn.utils.extmath import softmax


class RNN:

    def __init__(self, x_dim, y_dim, h_dim, t):
        self._U = np.random.uniform(low=-np.sqrt(1. / x_dim),
                                    high=np.sqrt(1. / x_dim),
                                    size=(h_dim, x_dim))
        self._V = np.random.uniform(low=-np.sqrt(1. / h_dim),
                                    high=np.sqrt(1. / h_dim),
                                    size=(y_dim, h_dim))
        self._W = np.random.uniform(low=-np.sqrt(1. / h_dim),
                                    high=np.sqrt(1. / h_dim),
                                    size=(h_dim, h_dim))
        self._h_dim = h_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self.T = t

    def forward(self,  x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        hidden_level = np.zeros((T + 1, self._h_dim))
        hidden_level[-1] = np.zeros(self._h_dim)
        # The outputs at each time step. Again, we save them for later.
        probability = np.zeros((T, self._y_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector.
            hidden_level[t] = np.tanh(self._U[:, x[t]] + self._W.dot(hidden_level[t - 1]))
            probability[t] = softmax(self._V.dot(hidden_level[t]))
        return [probability, hidden_level]

    def predict(self, x):
        prob, hidden_level = self.forward(x)
        return np.argmax(prob, axis=1)

    def backward(self):
        pass


if __name__ == '__main__':
    test = np.array([1, 2, 3])
    print(softmax(test))
