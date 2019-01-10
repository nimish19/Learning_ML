# map, reduce, filter and lambda 4
#find largest and smallest word
from functools import reduce
text = input("Enter the sentence: ").split()

lrg = list(filter(lambda x: len(x)>5, text))
print("Words whose length is gtreater than 5: ",lrg)

big = reduce(lambda x,y: max(x,y), text)
print("largest word in sentence: ",big)

small = reduce(lambda x,y: min(x,y), text)
print("the smallest word in sentence",small)