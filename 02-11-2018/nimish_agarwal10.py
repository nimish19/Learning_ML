ls = list(map(int,input("Enter numbers: " ).split(' ')))
total=0
for i in range(len(ls)):
    #check if ls[i] is 13 or the number just after 13
    if (ls[i-1]!=13 and ls[i]!=13):
        total += ls[i]
print (total)