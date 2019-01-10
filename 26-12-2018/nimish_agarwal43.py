#Submission 43
#import Automobile.csv file.
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Automobile.csv')
#Using MatPlotLib create a PIE Chart of top 10 car makers according to the number of their cars and explode the largest car maker
data = df['make'].value_counts().head(10)
plt.pie(data.head(10),explode=(0.2,0,0,0,0,0,0,0,0,0),labels=data.index,autopct="%2.2f%%")
plt.axis('equal')
plt.show()