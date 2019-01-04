#webscrapping using Selenium

from selenium import webdriver
import unicodedata
import pandas as pd

driver = webdriver.Chrome("F:/chromedriver.exe")
driver.get("https://www.icc-cricket.com/rankings/mens/team-rankings/odi")
correct_table = driver.find_element_by_class_name('table')

A=[]
B=[]
C=[]
D=[]
E=[]

for rows in correct_table.find_elements_by_tag_name('tr'):
    cells = rows.find_elements_by_tag_name('td')
    
    
    if len(cells)==5: 
        A.append(str(cells[0].text))
        B.append(str(cells[1].text))
        C.append(str(cells[2].text))
        D.append(str(cells[3].text))
        E.append(str(cells[4].text))
df = pd.DataFrame()
df['Rank'] = A
df['Team_name'] = B
df["Matches"] = C
df["Points"] = D
df["Rating"] = E
print(df)

driver.quit()