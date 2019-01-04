#submission 39
from collections import Counter 
import numpy as np
import matplotlib.pyplot as plt
#Create a random array of 40 integers from 5 - 15 using NumPy
arr2= np.random.randint(5,15,40)
#Find the most frequent value with and without Numpy
#with Numpy
count = np.bincount(arr2)
print('most frequent value with Numpy: ',np.argmax(count))

#withput Numpy
count = Counter(arr2)    
a=count.most_common(1)
print('most frequent without Numpy: ',a[0][0])