#Submission 13
#Enter the number of rows
n = int(input())

#prints pattern in shape of right angle triangle
for i in range(n):
    for j in range(0,i+1):
        print('* ',end='')  
    print('')
    
# inverted triangle
for i in reversed(range(n)):
    for k in range(i,0,-1):
        print('* ',end='')  
    print('')
