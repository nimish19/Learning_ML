#submission 38

import numpy as np
import matplotlib.pyplot as plt
# create data Centered around 150, Standard Deviation of 20, Total 1000 data points.
data = np.random.normal(150,20,1000)

plt.hist(data, 100)
plt.show()
print('Standard Deviation: ',data.std())
print('Variance: ',data.var())