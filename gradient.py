import numpy as np

def _gradi(f, x):
    h = 1e-4
    gradient = []
    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)
        x[idx] = temp_val - h
        fxh2 = f(x)
        val = (fxh1 - fxh2) / (2*h)
        gradient.append(val)
        x[idx] = temp_val
    return np.array(gradient, dtype='float64')


def gradi_2(fanc, x):
    if x.ndim == 1:
        return _gradi(fanc, x)
    else:
        gradient = np.zeros_like(x)
        for i, row in enumerate(x):
            gradient[i] = _gradi(fanc, row)
        
        return gradient