# map, reduce, filter and lambda 2
#find avg,min and max value
from functools import reduce 
data = list(map(int,input("Enter the values: ").split()))

avg = reduce(lambda x,y: x+y, data)/len(data)
print('average of given input: ',avg)
max = list(filter(lambda x: x>avg, data))
print("Values greater than average in list are: ",max)
min = list(filter(lambda x: x<avg, data))
print("Values smaller than average are: ",min)