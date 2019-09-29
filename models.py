import numpy as np
import common
import gradient

class twoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W = {}
        self.W['W1'] = np.random.rand(input_size, hidden_size)
        self.W['b1'] = np.zeros(hidden_size)
        self.W['W2'] = np.random.rand(hidden_size, output_size)
        self.W['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W = self.W
        z1 = np.dot(x, W['W1']) + W['b1']
        x2 = common.elu(z1)
        z2 = np.dot(x2, W['W2'])+ W['b2']
        return common.soft_max(z2)
    
#     tはone_hotの形式[0,0,0,1,0]の形式で受け取る
    def loss(self, x, t):
        y = self.predict(x)
        return common.cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        pred = self.predict(x)
        acc = np.array([pred.argmax(axis=1) == t.argmax(axis=1)], dtype='float64')
        return acc.sum() / x.shape[0]
    
    def decent(self, x, t):
        f = lambda a : self.loss(x, t)
        grads = {}
        grads['W1'] = gradient.gradi_2(f, self.W['W1'])
        grads['b1'] = gradient.gradi_2(f, self.W['b1'])
        grads['W2'] = gradient.gradi_2(f, self.W['W2'])
        grads['b2'] = gradient.gradi_2(f, self.W['b2'])
        return grads


class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = common.soft_max(z)
        return common.cross_entropy_error(y, t)
