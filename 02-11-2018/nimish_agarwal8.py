str = input("Enter a string: ")
Digits=0
Letters=0
for i in str:
    #check if i is digit
    if(i.isdigit()):
        Digits+=1
    if(i.isalpha()):
    #check if i is aplhabet
        Letters+=1
print("digits: ",Digits)
print('Letters: ',Letters)