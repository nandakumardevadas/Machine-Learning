import matplotlib.pyplot as plt
import numpy as np

greyhounds = 50
labrador = 50

greyhoundsHeight = 28 + 4 * np.random.randn(greyhounds)
labradorHeight = 24 + 4 * np.random.randn(labrador)
plt.hist([greyhoundsHeight, labradorHeight], stacked=True, color=['r', 'b'])
plt.show()