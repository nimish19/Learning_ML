#Submission 14

dictionary = {}

for i in range(3):
    keys = input("Key=")    #Input one key at  time
    values = int(input('value='))   #Input ine value at a time
    dictionary[keys] = values       #Assign key:value to a Dictionary
    
ignore = [13,14,17,18,19]
add = 0

for i in dictionary.values():
    if i not in ignore:     #selects values not in ignore list
        add += i