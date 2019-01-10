#submission 27

inp = list(map(int,input().split()))

def pallindrome(x):
    num = str(x)
    if num == num[::-1]:
        return True
    else:
        return False

print(all([i>0 for i in inp])and any([pallindrome(i) for i in inp]))