#submission pandas.extra.1
#Read data from all the year files starting from 1880 to 2017 
#add an extra column named as year that contains year of that particular data
import pandas as pd
data = pd.DataFrame()
ls = list(('yob'+str(i)+'.txt') for i in range(1880,2018))
j=1880

#Concatinate all the data to form single dataframe using pandas concat method
for i in ls:
    df=pd.read_csv(i,header=None,names=['Name','Sex','Number'])
    df['Year'] = j
    j+=1
    data = pd.concat([data,df],ignore_index=True,sort=True)

#Display the top 5 male and female baby names of 2017
name_2017 = df[df['Year']==2017]
male = name_2017[name_2017['Sex']=='M']
print('Top 5 Male names in 2017:\n\n',male.head())
female = name_2017[name_2017['Sex']=='F']
print('Top 5 Female names in 2017:\n\n',female.head())

#Calculate sum of the births column by sex as the total number of births in that year(use pandas pivot_table method)
births = pd.pivot_table(data,index=['Year','Sex'],aggfunc = np.sum)

#Plot the results of the above activity to show total births by sex and year
import matplotlib.pyplot as plt
#plots all data
births.plot()
#plots first 100 elements in data
births.head(100).plot()
