str=input('Enter a String: ')
str1=''
for i in str:
    str1+=i

str1=set(str1)
dict={}
for i in str1:
    dict1={i:str.count(i)}
    dict.update(dict1)

print(dict)