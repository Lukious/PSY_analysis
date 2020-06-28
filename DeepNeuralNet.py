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

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential

from keras.optimizers import SGD
from keras.optimizers import adam
from keras import metrics

from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization

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
  model = tf.keras.Sequential([
    layers.Dense(32, kernel_initializer='normal', activation = "relu", input_shape=(4,)), 
    layers.Dense(256, kernel_initializer='normal', activation = "relu"),
    layers.Dense(512, kernel_initializer='normal', activation = "relu"),
    layers.Dense(512, kernel_initializer='normal', activation = "relu"),
    layers.Dense(512, kernel_initializer='normal', activation = "relu"),
    layers.Dense(64, kernel_initializer='normal', activation = "relu"),
    layers.Dense(1, kernel_initializer='normal', activation = "relu"), 

  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  #opt = adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
  opt = RMSprop(lr=0.0001, decay=1e-6)

  model.compile(loss='mean_squared_error', 
              optimizer=opt, 
              metrics=[metrics.mse, metrics.mean_absolute_percentage_error])
  return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [decision]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$decision^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    # plt.show()
    plt.savefig('./DeepNet/MSEnMAE.png', dpi=300)
    
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
    
    sns_plot  = sns.pairplot(train_dataset[['decision','age','round','decision_time']], diag_kind="kde")
    sns_plot.savefig("./DeepNet/pairplot.png")

    train_stats = train_dataset.describe()
    train_stats.pop("decision")
    train_stats = train_stats.transpose()
    print(train_stats)
    
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
    
    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, batch_size=10, validation_split=0.2, verbose=0,callbacks=[PrintDot()])
    
    print(r2_score(train_labels, model.predict(normed_train_data)))

    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    
    '''
    plot_history(history)
    '''
    
    ##############################PLOT###############################
    '''
    plt.figure(figsize=(12, 4))
    plt.scatter(normed_train_data, train_labels, alpha=0.7, label='y_true')
    plt.scatter(normed_train_data, model.predict(normed_train_data), alpha=0.7, label='y_pred')
    plt.legend()
    plt.savefig('./DeepNet/MSEnMAE.png')
    plt.show()
    '''
    history.history.keys()
    
    val_loss_lst = history.history['val_loss']
    train_loss_lst = history.history['loss']
    
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(val_loss_lst)), val_loss_lst, label='val_loss')
    plt.plot(range(0, len(train_loss_lst)), train_loss_lst, label='train_loss')
    plt.legend()
    plt.savefig('./DeepNet/MSEnMAE.png')
    plt.show()
    ##############################PLOT###############################

    
    
    ##############################HISTORY###############################
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [decision]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$decision^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    # plt.show()
    plt.savefig('./DeepNet/MSEnMAE.png', dpi=300)
    ##############################HISTORY###############################

    hist.to_csv('./DeepNet/hist_DeepNet.csv',sep=',')
    loss, mae, mse  = model.evaluate(normed_test_data, test_labels, verbose=2)
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
    # plt.show()
    plt.savefig('./DeepNet/Predictions.png', dpi=300)