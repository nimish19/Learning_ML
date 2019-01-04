#submission 42

#Scrap the data from State/Territory and National Share (%) columns for top 6 states basis on National Share (%).
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import requests
url = 'https://en.wikipedia.org/wiki/List_of_states_and_union_territories_of_India_by_area'
source = requests.get(url).text
soup = BeautifulSoup(source)
all_tables = soup.find_all('table')
correct_table = soup.find('table',class_='wikitable')
A=[]
B=[]
for row in correct_table.find_all('tr'):
    cell = row.find_all('td')
    if(len(cell)==7):
        A.append(cell[1].find(text=True))
        B.append(cell[4].find(text=True))
df = pd.DataFrame()
df['state/teritory']=A
df['national_share']=B  
df= df.head(6)
# Create a Pie Chart using MatPlotLib and explode the state with largest national share %.
plt.pie(df['national_share'],explode=(0.2,0,0,0,0,0),labels=df['state/teritory'],autopct="%2.2f%%")
plt.axis('equal')
plt.show()