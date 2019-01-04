#submission 31
import re
data = input("Enter credit card no.: ")
valid = re.findall(r'[0-9a-zA-Z-_]+@\w+\.[a-z]{2,4}',data)
data = data.split()
for i in data:
    if i in valid:
        print(i)
