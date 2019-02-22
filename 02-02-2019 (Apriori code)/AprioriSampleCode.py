from efficient_apriori import apriori
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_csv('Market_Basket_Optimisation.csv')
df = df.replace(np.nan,'empty')

transactions=[]

for i in df.values:
    transactions.append(i)

itemsets,rules = apriori(transactions,min_support=0.003,min_confidence=0.2)
results = list(rules)