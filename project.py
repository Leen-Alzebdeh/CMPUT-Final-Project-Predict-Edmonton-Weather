# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import math 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sodapy import Socrata


def readData():
    
    client = Socrata("data.edmonton.ca",
                 app_token='WCNvl3VLVWa0jmIvmT3GBHd7a',
                 username="alzebdeh@ualberta.ca",
                 password="Dummypassword1!")
    
    #client = Socrata("data.edmonton.ca", None)

    limit = 75000

    # Returned as JSON from API / converted to Python list of
    # dictionaries by sodapy.
    results = client.get("s4ws-tdws", limit = limit)
    random.shuffle(results)
    
    # Sample:
    # {'row_id': '2779320211119', 'station_id': '27793', 'station_name': 'EDMONTON INTERNATIONAL CS', 
    #  'station_province': 'ALBERTA', 'station_latitude': '53.31', 'station_longitude': '-113.61', 
    #  'location': {'latitude': '53.31', 'longitude': '-113.61', 'human_address': '{"address": "", "city": "", "state": "", "zip": ""}'}, 
    #  'station_elevation_m': '715.00', 'station_climate_identifier': '3012206', 
    #  'station_wmo_identifier': '71155', 'station_tc_identifier': 'XEG', 'year': '2021', 
    #  'month': '11', 'day': '19', 'date': '2021-11-19T00:00:00.000', 
    #   'maximum_temperature_c': '-1.7', 
    #  'minimum_temperature_c': '-15.1', 'mean_temperature_c': '-8.4', 'heating_degree_days_c': '26.4', 
    #  'cooling_degree_days_c': '0.0', 'total_precipitation_mm': '0.6', 'snow_on_ground_cm': '11'} 

    data = []
    labels = []
    for result in results:
        try:
          month = result["month"]
          day = result["day"]
          latitude = result["station_latitude"]
          longitude = result["station_longitude"]
          elevation = result["station_elevation_m"]

          max_temp = result["maximum_temperature_c"]
          min_temp = result["minimum_temperature_c"]
          heating_degree_days_c = result["mean_temperature_c"]
          percipitation = result["total_precipitation_mm"]
          
          data.append([month, day, latitude, longitude, elevation])
          labels.append([max_temp, min_temp, heating_degree_days_c, percipitation])
  
        except Exception as e:
           pass
        
    data = np.array(data, dtype="float32")
    data = (data - np.mean(data)) / np.std(data)
    labels = np.array(labels, dtype="float32")
    labels = norm_denorm(labels)

    size = len(labels)

    train_data = np.array(data[0:int(size-0.30*size)+1], dtype='float32')
    valid_data = np.array(data[size- int(0.30*size + 1):int(size-0.10*size)+1], dtype='float32')
    test_data = np.array(data[int(size-0.10*size)+1:], dtype='float32')
   
    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1, dtype='float32')
    valid_data = np.concatenate(
        (np.ones([valid_data.shape[0], 1]),  valid_data), axis=1, dtype='float32')
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1, dtype='float32')
    
    train_labels = np.array(labels[0:int(size-0.30*size)+1], dtype='float32')
    valid_labels = np.array(labels[size- int(0.30*size + 1):int(size-0.10*size)+1], dtype='float32')
    test_labels = np.array(labels[int(size-0.10*size)+1:], dtype='float32')

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

def norm_denorm(y, norm=True):
  global mean, std
  if not ('mean' in globals()):
    mean = np.mean(y)
    std = np.std(y)
  
  # normalize features:
  if norm:
    y = (y - mean) / std
  # denorm
  else:
    y = y * std + mean
  return y

#################################
# Linear regression             

def linear_regression_predict(X, W, t=None):
    # X: Nsample x (d+1)
    # W: (d+1) x k

    t_hat = X @ W
    loss = (1/(2*t.shape[0]))*np.sum(np.square((t_hat-t))) + lamda*np.sum(np.transpose(W)@W)

    denorm_t_hat = norm_denorm(t_hat.copy(), False)
    t = norm_denorm(t.copy(), False)

    risk = mean_absolute_error(denorm_t_hat, t)

    return t_hat, None, loss, risk

def linear_regression_train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]

    train_losses = []
    valid_risk = []

    # initialization
    w = np.ones((X_train.shape[1], y_train.shape[1]), dtype='float32')
    # w: (d+1)

    W_best = None
    risk_best = 1000
    epoch_best = 0

    for epoch in range(MaxEpoch):
        #lr = alphas[0] * math.pow(0.5,  math.floor((1+epoch)/20))
        lr = alphas[0]
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_train[b*batch_size: (b+1)*batch_size, :] 
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, _, loss_batch , _ = linear_regression_predict(X_batch.copy(), w.copy(), y_batch.copy())
            
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent

            J = (1/(2*batch_size))*(np.transpose(X_batch) @ (y_hat_batch-y_batch)) + 2*lamda*np.sum(w)

            w = w - lr * J

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        train_loss = loss_this_epoch/batch_size
        train_losses.append(train_loss)
        # 2. Perform validation on the validation set by the risk
        _, _, _ , risk = linear_regression_predict(X_val.copy(),w.copy(),t_val.copy()) 
        valid_risk.append(risk)
        print('Epoch {}'.format(epoch))
        print('Accuracy/ Risk {}'.format(risk))
        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk < risk_best:
            epoch_best = epoch 
            W_best = w
            risk_best = risk

    # Return some variables as needed

    print("Best epoch", epoch_best)
    print("Best validation risk", risk_best)
    linear_regression_test(X_test,t_test, W_best, train_losses, valid_risk)

    return epoch_best, risk_best,  W_best, train_losses, valid_risk

def linear_regression_test(X_test, t_test, W_best, train_losses, valid_risk):
   
  
  # Report numbers and draw plots as required.
  plt.figure()
  plt.plot(range(MaxEpoch),train_losses, color="blue")
  plt.title('Training loss')
  plt.ylabel('loss')
  plt.xlabel('epochs')
  plt.savefig('losses_train_decay_lr_{}.jpg'.format(alpha))

  plt.figure()
  plt.plot(range(MaxEpoch),valid_risk, color="red")
  plt.title('validation risk')
  plt.ylabel('risk')
  plt.xlabel('epochs')
  plt.savefig('valid_decay_lr_{}.jpg'.format(alpha))

  _, _, _, risk_test = linear_regression_predict(X_test, W_best, t_test)
  print("Best test risk", risk_test)
  print("+--+"*40)
  print("+--+"*40)

##############################

def rnn_predict(X_train, t_train, X_val, t_val, X_test, t_test):

  X_train = norm_denorm(X_train, False)
  X_val = norm_denorm(X_val, False)
  X_test = norm_denorm(X_test, False)

  t_train = norm_denorm(t_train, False)
  t_val = norm_denorm(t_val, False)
  t_test = norm_denorm(t_test, False)
   
  hidden_units1 = 160
  hidden_units2 = 480
  hidden_units3 = 256
  drop = 0.3
  model = keras.Sequential([
    keras.layers.Dense(hidden_units1, kernel_initializer=tf.keras.initializers.HeUniform(), activation='relu'),
    keras.layers.Dropout(drop),
    keras.layers.Dense(hidden_units2, kernel_initializer=tf.keras.initializers.HeUniform(), activation='relu'),
    keras.layers.Dropout(drop),
    keras.layers.Dense(hidden_units3, kernel_initializer=tf.keras.initializers.HeUniform(), activation='relu'),
    keras.layers.Dropout(drop),
    keras.layers.Dense(4, kernel_initializer='normal', activation='linear')
  ])

  model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanAbsoluteError(),
    # List of metrics to monitor
    metrics=[keras.metrics.MeanSquaredError()],
  )
  print("Fit model on training data")
  history = model.fit(
      X_train,
      t_train,
      batch_size=64,
      epochs=50,
      # We pass some validation for
      # monitoring validation loss and metrics
      # at the end of each epoch
      validation_data=(X_val, t_val),
  ) 

# Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(X_test, t_test, batch_size=128)
  print("test risk, test error:", results)

  # summarize history for loss
  plt.figure()
  plt.plot(history.history['mean_squared_error'], color="red")
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend('train', loc='upper left')
  plt.savefig('neural_network_train_drop_{}.jpg'.format(drop))

  plt.figure()
  plt.plot(history.history['val_loss'], color="red")
  plt.title('model loss')
  plt.ylabel('risk')
  plt.xlabel('epoch')
  plt.legend('validation', loc='upper left')
  plt.savefig('neural_network_val_drop_{}.jpg'.format(drop))

#############################

def svr_regression_predit(X_train, t_train, X_val, t_val, X_test, t_test):


    regressor = SVR(kernel = 'sigmoid')
    svr = MultiOutputRegressor(regressor)
    svr.fit(X_train, t_train)

    #print("Validation data")
    #eval_svr(svr, X_val, t_val)

    print("Test data")
    eval_svr(svr, X_test, t_test)

def eval_svr(svr, X_test, t_test):
  from sklearn.model_selection import learning_curve

  # Generate predictions for testing data
  t_pred = svr.predict(X_test)

  # Evaluate the regressor
  mse_score = mean_squared_error(t_pred, t_test)
  print("MSE for regressors: ", mse_score)
  mae_score = mean_absolute_error(norm_denorm(t_pred, False), norm_denorm(t_test, False))
  print("MAE for regressors: ", mae_score)

  train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(svr, X_test, t_test, cv=30,return_times=True)

  plt.plot(train_sizes,np.mean(train_scores,axis=1))
  
  plt.savefig("svr.jpg")


##############################

# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readData()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)

alphas = [5e-1,1e-1, 1e-2, 1e-3, 1e-4]      # learning rate
batch_size = 16    # batch size
MaxEpoch = 100       # Maximum epoch
lambdas = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]         # weight decay
alpha = None
lamda = 0
risks = 1000
best_lambda = None

best_alpha = None

# linear regression
epoch_best, acc_best,  W_best, train_losses, valid_accs = linear_regression_train(X_train, t_train, X_val, t_val)


# rnn_predict(X_train, t_train, X_val, t_val, X_test, t_test)

# svr_regression_predit(X_train, t_train, X_val, t_val, X_test, t_test)
