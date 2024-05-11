# SiLu(Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def silu(x) : 
    return x * (1 / (1 + np.exp(-x))) # x * sigmoid(x)

def mish(x) : 
    return x * np.tanh(np.log(1 + np.exp(x)))

def relu(x) :
    return np.maximum(0,x)

def junglu(x) : 
    return (silu(x) + mish(x) + relu(x)) / 3

y = junglu(x)

plt.plot(x, y)
plt.grid()
plt.show()