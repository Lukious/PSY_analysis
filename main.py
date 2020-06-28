'''

Coded by Lukious
My psyside project

'''

import pandas as pd
import numpy as np

#from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)


def norm(x):
    #print(type(x))
    #print(type(train_stats['mean']))
    #print(type(train_stats['std']))
    return (x - train_stats['mean']) / train_stats['std']
    # return (x - np.mean(x)) / np.std(x)

def Preprocessing(data):
    data = data.dropna()
    print(data.shape[0])
    return data


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())+1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [decision]')
    plt.plot(hist['epoch'], hist['mae'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$decision^2$]')
    plt.plot(hist['epoch'], hist['mse'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()
    
if __name__ == '__main__': 
    
    raw_data = pd.read_csv("./data_ori.csv")
    clr_data = Preprocessing(raw_data)
    
    '''
    data = pd.DataFrame(clr_data[['fast','decision','age','gender']])
    label = pd.DataFrame(clr_data['ed'])
    '''
    
    label = pd.DataFrame(clr_data[['decision','age','round','decision_time']])
    data = pd.DataFrame(clr_data['decision_time'])
    
    train_dataset = label.sample(frac=0.8,random_state=0)
    test_dataset = label.drop(train_dataset.index)
    
    sns.pairplot(train_dataset[['decision','age','round','decision_time']], diag_kind="kde")
    
    train_stats = train_dataset.describe()
    train_stats.pop("decision")
    train_stats = train_stats.transpose()
    train_stats
    
    train_labels = train_dataset.pop('decision_time')
    test_labels = test_dataset.pop('decision_time')
    
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    
    

    ''' For multiple case
    scaler = StandardScaler()
    scaler.fit(train_labels)
    train_labels = train_labels.pop('decision_time')
    
    scaler = StandardScaler()
    scaler.fit(test_labels)
    test_labels = test_labels.pop('decision_time')
    '''
    
    model = build_model()
    model.summary()
    
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)
    
    #Set early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    EPOCHS = 100
    
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs): 
            if epoch % 100 == 0: print('')
            print('.', end='')

    
    print("Training!")
    
    history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,callbacks=[PrintDot()])
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    
    '''
    plot_history(history)
    '''
    
    ##############################HISTORY###############################
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [decision]')
    plt.plot(hist['epoch'], hist['mae'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$decision^2$]')
    plt.plot(hist['epoch'], hist['mse'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()
    ##############################HISTORY###############################


    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("테스트 세트의 평균 절대 오차: {:5.2f} decision".format(mae))

    
    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [decision]')
    plt.ylabel('Predictions [decision]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    
    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [decision]")
    _ = plt.ylabel("Count")
    plt.show()