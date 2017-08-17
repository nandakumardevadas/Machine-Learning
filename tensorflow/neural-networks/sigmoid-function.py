import matplotlib.pylab as plt
import numpy as np
x = np.arange(7, 8, 0.1)
y = 1 / (1+ np.exp(-x))
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()