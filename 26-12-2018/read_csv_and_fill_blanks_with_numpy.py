#read csv without Numpy
import numpy as np
import matplotlib.pyplot as plt

ndf = np.genfromtxt('Automobile.csv',delimiter=',',dtype=str)
df = pd.DataFrame(ndf)  
df.columns=ndf[0]
df=df.drop(index=0)
data =df['make'].value_counts().head(10)
#plot pie graph 
plt.pie(data.head(10),explode=(0.2,0,0,0,0,0,0,0,0,0),labels=data.index,autopct="%2.2f%%")
plt.axis('equal')
plt.show()
#fill missing values with 'nan' using Numpy
i,j=np.where(ndf=='')
index = list(zip(i,j))
for i in index:
    ndf[i]=np.nan