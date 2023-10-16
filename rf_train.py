import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
from itertools import repeat
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier


from sklearn.preprocessing import OneHotEncoder, normalize, LabelEncoder
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

m1 = open('data/match_1.json')
df1 = json.load(m1)

m2 = open('data/match_2.json')
df2 = json.load(m2)

df1 = pd.DataFrame(df1)
df2 = pd.DataFrame(df2)

df1[['label']] = df1[['label']].astype('category')
df2[['label']] = df2[['label']].astype('category')


df2 = df2.drop(df2[df2['label'] == 'no action'].index)

def flatten(l):
    return [item for sublist in l for item in sublist]

def flatten_column(df):
    label_ = []
    norm_ = []
    for i in range(len(df)):
        norm = df.iloc[i, 1]
        norm_.append(norm)
        df.iloc[i, 0] * len(norm)
        label_.append(list(repeat(df.iloc[i, 0], len(norm))))

    return flatten(label_), flatten(norm_)

lab1_, norm1_=flatten_column(df1)
lab2_, norm2_=flatten_column(df2)
df1_long=pd.DataFrame({'label':lab1_,'norm':norm1_})
df2_long=pd.DataFrame({'label':lab2_,'norm':norm2_})



# Creating a instance of label Encoder.
le = LabelEncoder()
label1 = le.fit_transform(df1_long['label'])

# log normalize
lognorm1_=np.log(norm1_)
df1_long['lognorm']=lognorm1_
df1_long['action']=label1

def reframed_data(dataset,serie_label,start_index, end_index, history_size):
    data = []
    labels_r = []
    labels_c=[]

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        reframed=np.append(dataset[indices],serie_label[indices[-1]])
        data.append(np.reshape(reframed, (history_size+1, 1)))
        labels_r.append(dataset[i])
        labels_c.append(serie_label[i])
    print('data size: ',np.array(data).shape)
    print('norm size: ', np.array(labels_r).shape)
    print('action size: ',np.array(labels_c).shape)
    return np.array(data), np.array(labels_r),np.array(labels_c)


# The model will be given the last 10 recorded to learn to predict the next time step.
history_size = 10

split_size = int(0.7* len(df1_long))
X_train, y_train_r,y_train_c= reframed_data(df1_long['lognorm'].values,
                                          df1_long['action'].values,
                start_index=0,
                end_index=split_size,
                history_size=history_size)

X_val, y_val_r,y_val_c= reframed_data(df1_long['lognorm'].values,
                                      df1_long['action'].values,
                start_index=split_size,
                end_index=int(0.85* len(df1_long)) ,
                history_size=history_size)

X_test, y_test_r,y_test_c= reframed_data(df1_long['lognorm'].values,
                                        df1_long['action'].values,
                start_index=int(0.85* len(df1_long)),
                end_index=None ,
                history_size=history_size)


print("##############################",'random forest', "#################################################")

X_train=np.squeeze(X_train)
y_train_r=np.squeeze(y_train_r)
y_train_c=np.squeeze(y_train_c)

X_val=np.squeeze(X_val)
y_val_r=np.squeeze(y_val_r)
y_val_c=np.squeeze(y_val_c)

X_test=np.squeeze(X_test)
y_test_r=np.squeeze(y_test_r)
y_test_c=np.squeeze(y_test_c)



rf_r = RandomForestRegressor(random_state=1234)
rf_r.fit(X_train, y_train_r)
y_predict_train_r=rf_r.predict(X_train)
y_predict_val_r=rf_r.predict(X_val)
y_predict_test_r=rf_r.predict(X_test)

print('Train  Score: %.2f MSE' % mean_squared_error(y_predict_train_r,y_train_r))
print('Val  Score: %.2f MSE' % mean_squared_error(y_predict_val_r,y_val_r))
print('Test  Score: %.2f MSE' % mean_squared_error(y_predict_test_r,y_test_r))

print('Train  Score: %.2f MAE' % mean_absolute_error(y_predict_train_r,y_train_r))
print('Val  Score: %.2f MAE' % mean_absolute_error(y_predict_val_r,y_val_r))
print('Test  Score: %.2f MAE' % mean_absolute_error(y_predict_test_r,y_test_r))


def plot_predict(y_real,y_predict,title):
    plt.figure(figsize=(25, 6))
    plt.plot(y_real, label='Actual')
    plt.plot(y_predict, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.xlim([0, len(y_real)])
    plt.title('lognorm_'+str(title))
    plt.savefig('figure/rf/'+str(title))
    plt.close()
    print('saved figure loss'+str(title))

plot_predict(y_real=y_val_r,
             y_predict=y_predict_val_r,
             title='validation set')

plot_predict(y_real=y_test_r,
             y_predict=y_predict_test_r,
             title='test set')

rf_c = RandomForestClassifier(random_state=1234)
rf_c.fit(X_train, y_train_c)

y_predict_train_c=rf_c.predict(X_train)
y_predict_val_c=rf_c.predict(X_val)
y_predict_test_c=rf_c.predict(X_test)

print('classification')
print('Train  Score: %.2f accuracy' % accuracy_score(y_predict_train_c,y_train_c))
print('Val  Score: %.2f accuracy' % accuracy_score(y_predict_val_c,y_val_c))
print('Test  Score: %.2f accuracy' % accuracy_score(y_predict_test_c,y_test_c))

y_predict_label=le.inverse_transform(y_predict_val_c)


y_predict_test = y_predict_test_c
y_predict_label = le.inverse_transform(y_predict_test)
y_predict_r = list(np.squeeze(np.exp(y_predict_test_r)))


def output_result(y_predict_r, y_predict_label,):
    norm = [y_predict_r[0]]
    norm_ = []
    action_ = []
    for i in range(1, len(y_predict_label)):
        if y_predict_label[i] == y_predict_label[i - 1]:
            norm.append(y_predict_r[i])

        else:
            norm_.append(norm)
            action_.append(y_predict_label[i - 1])
            norm = [y_predict_r[i]]

    s2 = pd.Series(norm_, name='norm')
    s1 = pd.Series(action_, name='action')
    df_res = pd.concat([s1, s2], axis=1)
    df_res.to_json('result/rf_prediction.json')
    return df_res

"""
df1['action_numeric']=le.fit_transform(df1['label'])
idx=list(df1['action_numeric'].value_counts().index)
df1['action_t+1']=df1['action_numeric'].shift(-1)
df1=df1[['action_numeric','action_t+1']].dropna(axis=0)


def find_action_proba(df, idx):
    dic = [0] * (len(idx))
    for i in range(len(idx)):
        dic[i] = defaultdict(int)

    for i in range(len(df)):
        dic[df.iloc[i, 0]][df.iloc[i, 1]] += 1

    # find the relevent proba
    proba = [0] * (len(idx))
    for i in range(len(idx)):
        proba[i] = np.cumsum([v / sum(dic[i].values()) for v in list(dic[i].values())])

    return proba, dic


def next_action(label_numeric, proba, dic, norm_catego, quantile_low_thred=5, quantile_high_thred=5):
    i = label_numeric
    rand = np.random.uniform()
    print(rand)

    def searchInsert(nums, target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (right - left) // 2 + left
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] == target:
                return mid
            else:
                right = mid - 1
        return left

    ind = searchInsert(proba[i], rand)

    action_next = list(dic[i].keys())[ind]

    quantile_low = [np.percentile(norm_catego[i], quantile_low_thred) for i in range(8)]
    quantile_high = [np.percentile(norm_catego[i], quantile_high_thred) for i in range(8)]

    return action_next, np.random.uniform(quantile_low[i], quantile_high[i])

"""