ls = list(map(int,input("Enter numbers: " ).split(',')))
big=small=int(ls[0])
sum=0
for i in ls:
    if(big < i):
        big = i
        
    if(small > i):
        small = i
ls.index(big)
del ls[ls.index(big)]
del ls[ls.index(small)]
for i in ls:
    sum+=int(i)
average=sum/len(ls)
print("Average:",int(average))