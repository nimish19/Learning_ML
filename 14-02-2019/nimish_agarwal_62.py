#submission 62
#from efficient_apriori import apriori
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("election_data.csv")

"""Considering the candidates who have contested in more than one Assembly elections,
Do such candidates contest from the same constituency in all the elections?
If not, does the change of constituency have any effect on the performance of the candidate?
Display the performance effect using a pie chart."""

#filter candidates who participated more than once
candidate = []
for i in df["Name_of_Candidate"].unique():
    temp = (df['Name_of_Candidate'][df["Name_of_Candidate"]==i].value_counts()) > 1
    if temp[0]:
        candidate.append(i)

#candidates who participated in different assembly        
assembly = []
for i in candidate:
    temp = (df['Assembly_no'][df["Name_of_Candidate"] == i].value_counts())
    if  temp.size>1:
        assembly.append([i,temp.size]) 

#candidates who participated in different constituency
constituency=[] 
for i in assembly:
    temp = (df['Constituency_no'][df["Name_of_Candidate"] == i[0]].value_counts())
    if temp.size>1:
        constituency.append([i[0],i[1],temp.size])

#dict of effect on performance by changing constituency
votes = []
for i in constituency:
    temp = df[df["Name_of_Candidate"] == i[0]]
    vote_count = []
    for j in temp['Constituency_no'].unique():
        vote_count.append(temp['Votes'][temp['Constituency_no'] == j].sum())
    if max(vote_count) != vote_count[0]:
        votes.append('Increased Performance')
    else:
        votes.append('Decreased Performance')

from collections import Counter
votes = Counter(votes)

#vishualizing using pie chart
plt.pie(votes.values(),labels=votes.keys(),autopct='%2.2f%%')
plt.axis('equal')
plt.title('Affect of changing Constituency')
plt.show()


'''Considering the candidates who have contested in more than one Assembly elections,
Do such candidates contest under the same party in all the elections? 
If not, how does the change in alliance of the candidate affect the outcome of the next election?
Display the outcome using a pie chart'''

#Candidates who changed Party
party = []
for i in assembly:
    temp = (df['Party'][df['Name_of_Candidate']==i[0]])
    if len(temp.unique())>1:
        party.append([i[0],len(temp.unique())])

#dict of effect on performance by changing Party
if len(party) == len(assembly):
    print("Yes, such candidates contest under the same party in all the elections.")
else:
    votes = []
    for i in party:
        temp = df[df["Name_of_Candidate"] == i[0]]
        vote_count = []
        for j in temp['Party'].unique():
            vote_count.append(temp['Votes'][temp['Party'] == j].sum())
        if max(vote_count) != vote_count[0]:
            votes.append('Increased Performance')
        else:
            votes.append('Decreased Performance')

from collections import Counter
votes = Counter(votes)

#vishualizing using pie chart
plt.pie(votes.values(),labels=votes.keys(),autopct='%2.2f%%')
plt.axis('equal')
plt.title('Affect of changing Party')
plt.show()

'''Do candidates who contested for multiple elections enjoy higher vote share percentages compared to
the candidates who have contested only once? Display the vote share percentage for both type of candidates
using a pie chart.'''

single_vote_share_perc = 0
single_candidate = []
for i in df.values:
    if i[6] not in candidate:
        single_candidate.append(i[6])
        single_vote_share_perc += i[-1]

single_vote_share_perc = single_vote_share_perc/len(single_candidate)

vote_share_perc = 0
candidate_count = 0 
for i in candidate:
    temp = df[df["Name_of_Candidate"] == i]
    vote_share_perc += temp['Vote_share_percentage'].sum()
    candidate_count += temp.shape[0] 
vote_share_perc = vote_share_perc/candidate_count   

votes = {'single_vote_share_perc' : single_vote_share_perc, 'vote_share_perc' : vote_share_perc}
    
#vishualizing using pie chart
plt.pie(votes.values(),labels=votes.keys(),autopct='%2.2f%%')
plt.axis('equal')
plt.title('Affect')
plt.show()