from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from numpy.random import seed
from math import sqrt
import numpy as np # linear algebra
import pandas as pd # data processing
from datetime import datetime
import timeit
from utils.dataloader import StockDataset
from utils.evaluate_model import EvaluateModel
import torch
import tqdm
import pickle
import tensorflow as tf
import os
import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from torch.utils.tensorboard import SummaryWriter

# Create the tensorboard writer
log_folder = f"runs/" + "sklearn_models"
writer = SummaryWriter(log_folder)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_folder, histogram_freq=1)

def GB(xtrain, xtest, ytrain, ytest):
    # Add Tensorboard
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0).fit(xtrain, ytrain,)
    cls_proba = model.predict_proba(xtest)
    cls_proba = cls_proba[:,1]
    acc = accuracy_score(ytest, np.around(cls_proba))
    return model, acc

def LR(xtrain, xtest, ytrain, ytest):
    model = linear_model.LogisticRegression(random_state=None, max_iter=max_iter).fit(xtrain, ytrain,)
    cls_proba = model.predict_proba(xtest)
    cls_proba = cls_proba[:,1]
    acc = accuracy_score(ytest, np.around(cls_proba))
    return model, acc

def SVM(xtrain, xtest, ytrain, ytest):
    model = svm.SVC(kernel = 'rbf',gamma=GAMMA,probability=True).fit(xtrain, ytrain,) # SVM with rbf Kernel Functions.fit(xtrain, ytrain)
    cls_proba = model.predict_proba(xtest)
    cls_proba = cls_proba[:,1]
    acc = accuracy_score(ytest, np.around(cls_proba))
    return model, acc

# Set the seed for reproducibility
seed(69)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Set the hyperparameters
GAMMA = 10              # for SVM model
max_iter = 5000         # for Logistic regression model
n_estimators = 100      # for Gradient boosting model
learning_rate = 0.001   # for Gradient boosting model
sequence_length = 5

symbol_list = ['GOOG', 'AMZN', 'BLK', 'IBM', 'AAPL']
test_list = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
model_list = {
    'GB': [],
    'LR': [],
    'SVM':[]
}

test_set_accuracy = {
    'GB': [],
    'LR': [],
    'SVM':[]
}

test_sets = 6
for i in range(test_sets):
    print(f'testing set {i}')
    symbol_accuracy = {
        'GB': [],
        'LR': [],
        'SVM':[] 
    }
    # for s in symbol:
    # Load the data
    for _,symbol in tqdm.tqdm(enumerate(symbol_list), total=len(symbol_list)):
        train_data = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=True, normalize=False, offset=0)
        x_train_temp = np.delete(train_data.train_set, 15, 1)
        x_train_temp = np.delete(x_train_temp, 15, 1)
        x_train = x_train_temp[:, 0:17]
        y_train = x_train_temp[:, 18]
    
        test_data = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=False, normalize=False, offset=i)
        x_test_temp = np.delete(test_data.test_set, 15, 1)
        x_test_temp = np.delete(x_test_temp, 15, 1)
        x_test = x_test_temp[:, 0:17]
        y_test = x_test_temp[:, 18]
        
        # Create the models
        model, accuracy = GB(x_train, x_test, y_train, y_test)
        filename = f'./weights/GB_{symbol}.sav'
        pickle.dump(model, open(filename, 'wb'))
        symbol_accuracy['GB'].append(accuracy)
        model_list['GB'].append(model)
        
        model, accuracy = LR(x_train, x_test, y_train, y_test)
        filename = f'./weights/LR_{symbol}.sav'
        pickle.dump(model, open(filename, 'wb'))
        symbol_accuracy['LR'].append(accuracy)
        model_list['LR'].append(model)
        
        model, accuracy = SVM(x_train, x_test, y_train, y_test)
        filename = f'./weights/SVM_{symbol}.sav'
        pickle.dump(model, open(filename, 'wb'))
        symbol_accuracy['SVM'].append(accuracy)
        model_list['SVM'].append(model)
        
    test_set_accuracy['GB'].append(np.array(symbol_accuracy['GB']).mean())
    test_set_accuracy['LR'].append(np.array(symbol_accuracy['LR']).mean())
    test_set_accuracy['SVM'].append(np.array(symbol_accuracy['SVM']).mean())
    
# Plot the losses
plt.figure(1)
plt.plot(test_list, test_set_accuracy['GB'],  label='Gradient Boosting', c="b", lw=2)
plt.scatter(test_list, test_set_accuracy['GB'], c="b", lw=2)
plt.plot(test_list, test_set_accuracy['LR'], label='Logistic Regression', c="r", lw=2)
plt.scatter(test_list, test_set_accuracy['LR'], c="r", lw=2)
plt.plot(test_list, test_set_accuracy['SVM'], label='SVM', c="g", lw=2)
plt.scatter(test_list, test_set_accuracy['SVM'], c="g", lw=2)
plt.title('Models Accuracy for Different testing sets')
plt.xlabel('Testing set')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'plots/New_eval.png')

# Save the test set accuracy
with open('plots/test_set_accuracy.json', 'w') as fp:
    json.dump(test_set_accuracy, fp)

writer.add_figure('New_eval', plt.gcf())