#submission 30
from bs4 import BeautifulSoup
import pandas as pd
import urllib

icc = urllib.request.urlopen("https://www.icc-cricket.com/rankings/mens/team-rankings/odi")
soup = BeautifulSoup(icc)
all_tables = soup.find_all('table')
correct_table = soup.find('table',class_="table")
#Creating links
A=[]
B=[]
C=[]
D=[]
E=[]
for row in correct_table.findAll('tr'):
    cell = row.find_all('td')
    state = row.find_all('tr')
    if(len(cell)==5):
        A.append(cell[0].find(text=True))
        B.append(cell[1].find(text=True))
        C.append(cell[2].find(text=True))
        D.append(cell[3].find(text=True))        
        E.append(cell[4].find(text=True))

df = pd.DataFrame()
df['Rank'] = A
df['Team_name'] = B
df["Matches"] = C
df["Points"] = D
df["Rating"] = E
print(df)