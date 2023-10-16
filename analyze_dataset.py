import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
from itertools import repeat
from sklearn.preprocessing import OneHotEncoder,normalize,LabelEncoder

m1=open('data/match_1.json')
df1 = json.load(m1)

m2=open('data/match_2.json')
df2 = json.load(m2)

df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)

df1[['label']]=df1[['label']].astype('category')
df2[['label']]=df2[['label']].astype('category')


##Pie figure


plt.figure(figsize=(30,15))

plt.subplot(1,2,1)
category = df1.label.value_counts()
plt.title("match1")
plt.pie(x=category.values, labels=category.index, counterclock=False, startangle=90)
plt.legend(category.index)

plt.subplot(1,2,2)
category = df2.label.value_counts()
plt.title("match2")
plt.pie(x=category.values, labels=category.index, counterclock=False, startangle=90)
plt.legend(category.index)
plt.savefig('figure/analyze/compare_action.png')
print('action pie comparison is saved')

df2 = df2.drop(df2[df2['label'] == 'no action'].index) # remove missing values

def flatten(l):
    return [item for sublist in l for item in sublist]

idx1=list(df1['label'].value_counts().index)
idx1


def flatten_column(df):
    label_ = []
    norm_ = []
    for i in range(len(df)):
        norm = df.iloc[i, 1]
        norm_.append(norm)
        df.iloc[i, 0] * len(norm)
        label_.append(list(repeat(df.iloc[i, 0], len(norm))))

    return flatten(label_), flatten(norm_)

lab1_,norm1_=flatten_column(df1)
lab2_,norm2_=flatten_column(df2)
df1_long=pd.DataFrame({'label':lab1_,'norm':norm1_})
df2_long=pd.DataFrame({'label':lab2_,'norm':norm2_})


#Hist figure

norm1_catego_ = []
norm2_catego_ = []

f1 = plt.figure(figsize=(45,20))

for i, v in enumerate(idx1):
    norm1_cat = df1_long[df1_long['label'] == v]['norm']
    norm2_cat = df2_long[df2_long['label'] == v]['norm']
    norm1_catego_.append(norm1_cat.values)
    norm2_catego_.append(norm2_cat.values)

    f1.add_subplot(2, 4, i + 1)
    sns.histplot(x=norm1_cat, kde=True, stat='density', color='red', alpha=0.4).set(title=v)
    sns.histplot(x=norm2_cat, kde=True, stat='density', color='green', alpha=0.4).set(title=v)

plt.savefig('figure/analyze/compare_hist.png')
print('hist comparison is saved')

#boxplot
df1_long['match']=1
df2_long['match']=2
df_total=pd.concat([df1_long, df2_long], ignore_index=True, axis=0)

plt.figure(
    figsize=(25,15)
)
sns.boxplot(data=df_total,x='norm',y='label',hue='match')
plt.grid()
plt.savefig('figure/analyze/compare_boxplot.png')
print('boxplot comparison is saved')

plt.figure(
    figsize=(40,10)
)
plt.subplot(2,1,1)
plt.plot(norm1_)
plt.xlim(0,len(norm1_))
plt.ylim(0,800)
plt.title('match_1')

plt.subplot(2,1,2)
plt.plot(norm2_)
plt.xlim(0,len(norm2_))
plt.ylim(0,800)
plt.title('match_2')
plt.savefig('figure/analyze/movement.png')
print('acceleration by timestep is saved')


