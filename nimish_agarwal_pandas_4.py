#submission pandas.extra.4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

ls = []    
with open('bitly-usagov-example.txt') as fp:
    for line in fp:
        ls.append(json.loads(line))
df = pd.DataFrame(ls)
#Replace the 'nan' values with 'Mising' and ' ' values with 'Unknown' keywords
df = df.fillna('Mising')
df = df.replace(to_replace='',value='Unknown')  
#Print top 10 most frequent time-zones from the Dataset i.e. 'tz', 
#with pandas
print(df['tz'].value_counts().head(10))

#without pandas
from collections import Counter
time_zone = Counter(df['tz'])
time_zone = time_zone.most_common(10)

#Count the number of occurrence for each time-zone
freq_time_zone = df['tz'].value_counts()

#Plot a bar Graph to show the frequency of top 10 time-zones (using Seaborn)
import seaborn as sb
graph_time_zone = sb.barplot(freq_time_zone.head(10).index, freq_time_zone.head(10))
for item in graph_time_zone.get_xticklabels():
    item.set_rotation(90)

#From field 'a' which contains browser information and separate out browser capability(i.e. the first token in the string eg. Mozilla/5.0)
#capabilities = pd.DataFrame(df['a']).filter(regex='^([a-z,A-Z]+/\d+\.\d+)$',axis=1)

ls = []
for i in df['a']:
    i=i.split('(',1)
    ls.append(i[0])   
capabilities = pd.DataFrame(ls) 
capabilities = capabilities[0].value_counts()

#Count the number of occurrence for separated browser capability field and plot bar graph for top 5 values (using Seaborn)
graph_capability = sb.barplot(capabilities.head().index,capabilities.head())
for item in graph_capability.get_xticklabels():
    item.set_rotation(35)

#Add a new Column as 'os' in the dataset, separate users by 'Windows' for the values in  browser information column i.e. 'a' that contains "Windows" and "Not Windows" for those who don't
k=df["a"].str.find("Windows")
df["os"]=np.where(k==-1,'Not Windows','Windows')