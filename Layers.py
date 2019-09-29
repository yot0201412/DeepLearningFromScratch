
# coding: utf-8

# In[1]:


import numpy as np
import common as co


# In[2]:


class MulLayer():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self, dout):
        dout_x = dout * self.y
        dout_y = dout * self.x
        return dout_x, dout_y


# In[9]:


class AddLayer():
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x + self.y

    def backward(self, out):
        return out * 1, out * 1


# In[14]:


class Relu():
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[mask] = 0
        return out

class affine():
    def __init(self, W, b):
        self.W = W
        self.b = b
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
    
# In[ ]:


class SoftMax_with_loss():
    def __init__(self):
        self.t = None
        self.y = None
        self.loss = None

    def forward(self, a, t):
        self.y = co.softmax(a)
        self.t = t
        self.loss = co.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        bp = (self.y - self.t) / batch_size
        return bp
