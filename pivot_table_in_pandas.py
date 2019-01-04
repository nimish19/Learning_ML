#pivot table in pandas
import pandas as pd
import numpy as np
 
#Create a DataFrame
d = {
    'Name':['Alisa','Bobby','Cathrine','Alisa','Bobby','Cathrine',
            'Alisa','Bobby','Cathrine','Alisa','Bobby','Cathrine'],
    'Exam':['Semester 1','Semester 1','Semester 1','Semester 1','Semester 1','Semester 1',
            'Semester 2','Semester 2','Semester 2','Semester 2','Semester 2','Semester 2'],
     
    'Subject':['Mathematics','Mathematics','Mathematics','Science','Science','Science',
               'Mathematics','Mathematics','Mathematics','Science','Science','Science'],
   'Score':[62,47,55,74,31,77,85,63,42,67,89,81],
   'marks':[62,47,55,74,31,77,85,63,42,67,89,81]}
 
df1 = pd.DataFrame(d,columns=['Name','Exam','Subject','Score','marks'])

pd.pivot_table(df1, index=['Exam','Subject'], aggfunc=np.sum)

pd.pivot_table(df1, index=['Name','Subject'], aggfunc='sum')

pd.pivot_table(df1, index=['Exam','Subject'], aggfunc='count')