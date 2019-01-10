#tutorial Counter, OrderedDict, namedtuple, defaultdict, any, all
a = "www.google.com"

from collections import Counter
#counter counts all the number of times letters are rpeated in dict
new = Counter(a)

print (new.most_common(3))

from collections import OrderedDict
#sets a default order to the items in dictionary
od = OrderedDict()

od["Name"] = "Kunal"
od["Place"] = "Jaipur"
od["Area"] = "Sitapura"

print (od)


from collections import namedtuple
#sets a new datatype in tuple for reference 
maruti = namedtuple("Car","Model,Price,Color,CC")

wagonR = maruti("Hatchback",150000,"Metallic",555)

Swift = maruti("Hatchback",250000,"White",650)

from collections import defaultdict
#initializes keys with default values of i.e., defaultdict(int,{})
dd = defaultdict(int)
for i in a:
    dd[i] += 1

l = [8,4,6,8,4,6,9,7,4,-2]
#all is similar to and
#any is similar to or
print(all([i>0 for i in l]))