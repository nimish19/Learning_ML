## map, reduce, filter and lambda 3
#list of tuple with temperature in Celcius
temps = [('India',29),('China',32),('Korea',25),('Nepal',30)]
#creating list of tuples with temperature in farenhiet
farh = [(x[0],((9/5)*x[1])+32) for x in temps]

print(farh)