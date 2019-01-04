#Submission 15

#declare A list of elements 1 to 50
lst = list(range(1,51)) 

for i in lst:
    #check divisibility by 3 and 5
    if(i%3==0 and i%5==0): 
        print('FizzBuzz')
    #check divisibility by 5
    elif(i%5==0):   
        print("Buzz")
    #check dividibility by 3
    elif(i%3==0):  
        print("Fizz")
    else:
        print(i)