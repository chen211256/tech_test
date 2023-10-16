import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
from itertools import repeat
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.metrics import mean_squared_error,mean_absolute_error
# LSTM for international airline passengers problem with memory
import tensorflow as tf
from keras.models import Sequential

from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional

from sklearn.preprocessing import OneHotEncoder, normalize, LabelEncoder

from keras import callbacks
import os
script_dir = os.path.dirname(__file__)


m1 = open(os.path.join(script_dir ,'data/match_1.json'))
df1 = json.load(m1)


df1 = pd.DataFrame(df1)


df1[['label']] = df1[['label']].astype('category')


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

df1_long=pd.DataFrame({'label':lab1_,'norm':norm1_})


tf.random.set_seed(1234)

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
                end_index=int(0.9* len(df1_long)) ,
                history_size=history_size)

X_test, y_test_r,y_test_c= reframed_data(df1_long['lognorm'].values,
                                        df1_long['action'].values,
                start_index=int(0.9* len(df1_long)),
                end_index=None ,
                history_size=history_size)



tf.random.set_seed(1234)
model_reg = Sequential()
model_reg.add(LSTM(50, activation='relu',input_shape=X_train.shape[-2:],return_sequences = True))
model_reg.add(LSTM(25,activation='relu'))
model_reg.add(Dense(1))
model_reg.compile(loss='mae', optimizer='adam')


earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)


# fit network
epochs=10
history_reg = model_reg.fit(X_train,y_train_r, epochs=epochs, batch_size=72, validation_data=(X_val,y_val_r),
    validation_steps=10,callbacks=[earlystopping])

model_reg.save(os.path.join(script_dir ,"result/lstm/lstm_reg_steps_"+str(epochs)+"_epochs.h5"))
print("saved regression model to disk")
model_reg.summary()




train_loss = history_reg.history['loss']
val_loss = history_reg.history['val_loss']

# Plot loss
def plot_loss(train_loss,val_loss,epochs,title):
    epochs_range = range(epochs)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss_'+str(title))
    plt.savefig('figure/lstm/loss_'+str(title)+'.png')
    plt.close()
    print('saved figure loss')


plot_loss(train_loss = train_loss,
          val_loss= val_loss,
          epochs=epochs,
          title='mae')


def plot_predict(y_real,y_predict,title):
    plt.figure(figsize=(25, 6))
    plt.plot(y_real, label='Actual')
    plt.plot(y_predict, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.xlim([0, len(y_real)])
    plt.title('lognorm_'+str(title))
    plt.savefig('figure/lstm/lstm_'+str(title)+"_steps_"+str(epochs)+"_epochs.png")
    plt.close()
    print('saved figure loss'+str(title))

y_predict_train_r=model_reg.predict(X_train)
y_predict_val_r=model_reg.predict(X_val)
y_predict_test_r=model_reg.predict(X_test)

plot_predict(y_real=y_val_r,
             y_predict=y_predict_val_r,
             title='validation set')

plot_predict(y_real=y_test_r,
             y_predict=y_predict_test_r,
             title='test set')

trainScore = mean_squared_error(y_train_r, y_predict_train_r[:,0])
print('Train  Score: %.2f MSE' % (trainScore))
valScore = mean_squared_error(y_val_r, y_predict_val_r[:,0])
print('Val Score: %.2f MSE' % (valScore))
testScore = mean_squared_error(y_test_r, y_predict_test_r[:,0])
print('Test Score: %.2f MSE' % (testScore))

trainScore = mean_absolute_error(y_train_r, y_predict_train_r[:,0])
print('Train  Score: %.2f MAE' % (trainScore))
valScore = mean_absolute_error(y_val_r, y_predict_val_r[:,0])
print('Val Score: %.2f MAE' % (valScore))
testScore = mean_absolute_error(y_test_r, y_predict_test_r[:,0])
print('Test Score: %.2f MAE' % (testScore))



tf.random.set_seed(1234)
model_cl = Sequential()
model_cl.add(LSTM(50, activation='relu',input_shape=X_train.shape[-2:],return_sequences = True))
model_cl.add(LSTM(25,activation='relu'))
model_cl.add(Dense(8,activation='softmax'))
model_cl.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


# fit network
history_cl = model_cl.fit(X_train,
                       y_train_c,
                       epochs=epochs,
                       batch_size=72,
                       validation_data=(X_val,y_val_c),
                       validation_steps=10,
                       callbacks=[earlystopping])

model_cl.save(os.path.join(script_dir ,"result/lstm/lstm_class_steps_"+str(epochs)+"_epochs.h5"))
print("saved classification model to disk")
model_cl.summary()

train_loss = history_cl.history['loss']
val_loss = history_cl.history['val_loss']
epochs_range = range(epochs)

plot_loss(train_loss = train_loss,
          val_loss= val_loss,
          epochs=epochs,
          title='crossval')

y_predict_train_c=model_cl.predict(X_train)
y_predict_val_c=model_cl.predict(X_val)
y_predict_test_c=model_cl.predict(X_test)


trainScore = accuracy_score(y_train_c, y_predict_train_c.argmax(axis=1))
print('Train  Score: %.2f Acccuracy' % (trainScore))
valScore = accuracy_score(y_val_c, y_predict_val_c.argmax(axis=1))
print('Val Score: %.2f Accuracy' % (valScore))
testScore = accuracy_score(y_test_c, y_predict_test_c.argmax(axis=1))
print('Test Score: %.2f Accuracy' % (testScore))

y_predict_test = y_predict_test_c.argmax(axis=1)
y_predict_label = le.inverse_transform(y_predict_test)
y_predict_r = list(np.squeeze(np.exp(y_predict_test_r)))


def output_result(y_predict_r, y_predict_label, epochs):
    norm = [y_predict_r[0]]
    norm_ = []
    action_ = []
    for i in range(1, len(y_predict_test)):
        if y_predict_label[i] == y_predict_label[i - 1]:
            norm.append(y_predict_r[i])

        else:
            norm_.append(norm)
            action_.append(y_predict_label[i - 1])
            norm = [y_predict_r[i]]

    s2 = pd.Series(norm_, name='norm')
    s1 = pd.Series(action_, name='action')
    df_res = pd.concat([s1, s2], axis=1)
    df_res.to_json(os.path.join(script_dir ,'result/lstm_prediction_' + str(epochs) + '.json'))
    return df_res

df=output_result(y_predict_r,y_predict_label,epochs)


"""
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