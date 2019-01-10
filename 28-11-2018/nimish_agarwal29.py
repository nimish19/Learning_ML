#submission 29
import re
data = input("Enter the expressions: ")
result = re.findall(r'[+-.]?\d+\.\d+', data)
#create list of items in data
data = data.split()
for i in range (len(data)):
    #check if i in data is present in result
    print(data[i] in result)
    
    
