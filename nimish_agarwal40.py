#submission 40
import numpy as np
#given 9 space separated numbers
num = '6 9 2 3 5 8 1 5 4'
num = num.split()

#convert it into a 3x3 NumPy array of integers
x = np.array(list(int(i) for i in num),ndmin=2)
x = np.reshape(x,(3,3))
print(x)