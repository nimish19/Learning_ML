#submission 25
l1=[]

while True:
    inp = input()
    if inp == '' :
        break
    inp = tuple(inp.split(","))
    #append tuple as element in list
    l1.append(inp)
#using list comprehension to typecast str value to integer value
l1=[(x[0],int(x[1]),int(x[2])) for x in l1]
l1.sort()
print('Sorted list: ',l1)