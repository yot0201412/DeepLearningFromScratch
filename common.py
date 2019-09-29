import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def elu(x):
    return np.where( x > 0, x , np.exp(x) -1)

def soft_max(list):
    max_list = list.max()
    list_n = list - max_list
    exps = np.exp(list_n)
    sum_exp = exps.sum()
    y = exps / sum_exp
    return y

def mean_squared_error(predict, y):
    dif = predict - y
    return 0.5 * (dif ** 2).sum()

def cross_entropy_error(predict, t):
    if predict.ndim == 1:
        predict.reshape(1, predict.size)
        t.reshape(1, t.size)
    delta = 1e-7
    return -np.sum( t * np.log(predict + delta)) / predict.shape[0]

def gradi(f, x):
    h = 1e-4
    grad = []
    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)
        x[idx] = temp_val - h
        fxh2 = f(x)
        val = (fxh1 - fxh2) / (2*h)
        grad.append(val)
        x[idx] = temp_val
    return np.array(grad, dtype='float64')

def And(input):
    s = -0.7
    w = np.array([0.5, 0.5])
    output = np.sum(input * w) + s
    if output >= 0:
        return 1
    else:
        return 0
    
def Nand(input):
    s = 0.7
    w = np.array([-0.5, -0.5])
    output = np.sum(input * w) + s
    if output < 0:
        return 0
    else:
        return 1
    
def Or(input):
    s = -0.2
    w = np.array([0.5, 0.5])
    output = np.sum(input * w) + s
    if output > 0:
        return 1
    else:
        return 0
    
def Xor(input):
    if Or(input) and Nand(input):
        return 1
    else:
        return 0