#Submission 18

item = input("Please enter the text: ").strip(',').strip('.')
item = item.replace(' ','').lower()
l1=[]
l1 = [_ for _ in item if _ not in l1]
l1.sort()
l2 = set(l1)
if(len(l2) == 26):
    print('PANGRAM')
else:
    print('NOT PANGRAM')