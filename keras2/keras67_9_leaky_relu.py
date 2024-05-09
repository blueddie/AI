import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def leakyReLu(x) :
    return np.maximum(0.1 * x, x)

y = leakyReLu(x)

plt.plot(x, y)
plt.grid()
plt.show()