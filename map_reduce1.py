# map, reduce, filter and lambda 1

#list of radius
radius = list(map(int,input('Enter the radiuses: ').split()))
#import math for using value of pi
import math
#map area of cirlce for value in radius
area_circle = list(map(lambda x: math.pi*(x**2), radius))
print('area of circle for entered list of raduius are: ', area_circle)
