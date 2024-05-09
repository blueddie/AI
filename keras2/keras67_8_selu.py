import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5, 5, 0.1)
lambda_ = 1.0507
alpha_ = 1.67326

def selu(x, lambda_=1.0507, alpha_=1.67326):
    return lambda_ * np.where(x > 0, x, alpha_ * (np.exp(x) - 1))

y = selu(x)
plt.plot(x, y)
plt.title("SELU Activation Function")
plt.xlabel("x")
plt.ylabel("SELU(x)")
plt.grid(True)
plt.show()