import numpy as np
import math
import matplotlib.pyplot as plt

size = 1000

size1 = math.floor(size / 2)
size2 = math.ceil(size / 2)

x1_banana1 = np.random.uniform(-2, 6, size1)
x2_banana1 = -0.3 * (x1_banana1-2)**2 + 4 + 0.7 * np.random.randn(size1)

x1_banana2 = np.random.uniform(-6, 2, size2)
x2_banana2 = np.random.randn(size2)


plt.scatter(x1_banana1, x2_banana1, color='blue', marker='o')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')

plt.show()
