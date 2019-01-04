#Submission 11

def Add(*args):
    '''Adds all the elements in args[0]'''
    addition=0
    for items in args[0]:
        addition+= items
    return addition

def Multiply(*args):
    '''Multipy all emlements in args[0]'''
    product=1
    for items in args[0]:
        product*=items
    return product

def Largest(*args):
    '''find the largest using max()'''
    return max(args[0])

def Smallest(*args):
    '''find the smallest using min()'''
    return min(args[0])

def Sorted(*args):
    '''sorting using sorted()'''
    return sorted(args[0])

def Remove_Duplicates(*args):
    '''Remove double occurense of elements'''
    new_list=[]
    for items in args[0]:
        if items not in new_list:
            new_list.append(items)
    return sorted(new_list)

# Takes input in list format of integer type
numbers=list(map(int,input().split()))
#call Add(*args) Func.
print("Sum = ",Add(numbers))
#calls Multipy(*args) Func.
print("Multiply = ",Multiply(numbers))
#calls Largest(*args) Func.
print("Largest = ",Largest(numbers))
#calls Smallest(*args) Func.
print("Smallest = ",Smallest(numbers))
#calls Sorted(*args) Func.
print("Sorted = ",Sorted(numbers))
#calls Remove_Duplicates(*args) Func.
print("Without Duplicates = ",Remove_Duplicates(numbers))
