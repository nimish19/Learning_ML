#submission37

import numpy as np
import matplotlib.pyplot as plt
incomes = np.random.normal(100.0, 20.0, 10000)
print(incomes)

plt.hist(incomes,50)
plt.show()
print('Mean of Incomes: ',np.mean(incomes))
print("Median of Incomes: ",np.median(incomes))
incomes=np.append(incomes,[100000])
print(np.mean(incomes))
print(np.median(incomes))
