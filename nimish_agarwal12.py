#Submission 12

def bricks(lst):
    s,b,g = lst
    # checks if number of smaller bricks is sufficient for the wall
    if g%5 > s: 
        return False
    # checks if total length of available bricks is sufficient for building the wall
    if s+b*5 >= g:
        return True
    else:
        return False

inp = list(map(int,input().split(",")))
print (bricks(inp))

'''
length_wall=list(map(int,input().split(',')))
width = 0
small_b=1
big_b=5

def small_brick():
    global width
    width += small_b

def big_brick():
    global width
    width += big_b
    
def less_than_big_b():
    for i in range(length_wall[0]):
        if(width==length_wall[2]): break
        small_brick()

def more_than_big_b():
    for i in range(length_wall[1]):
        if(width==length_wall[2]): break
        big_brick()
        if(width<=2*big_b):
            less_than_big_b()
    less_than_big_b()
        
if(length_wall[2]<=small_b):    
    less_than_big_b()    
    
if(length_wall[2]>=big_b):
    more_than_big_b()
    
print(width == length_wall[2])
'''