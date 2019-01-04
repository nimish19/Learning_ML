#submission 41

import numpy as np
import pandas as pd
# Read the Automobile.csv file and perform the following task :
data = pd.read_csv("Automobile.csv")
# 1. Handle the missing values for Price column
data['price'] = data['price'].fillna(data['price'].mean())
# 2. Get the values from Price column into a numpy.ndarray
array = np.array(data['price'])
# 3. Calculate the Minimum Price, Maximum Price, Average Price and Standard Deviation of Price
print('Average price: ',array.mean())
print('Minimum price: ',np.amin(array))
print('Maximum price: ',np.amax(array))
print('Standard Deviation: ',array.std())