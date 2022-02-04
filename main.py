import numpy as np
import pandas as pd
from components.preprocess import preprocess
from components.model import model
from components.dataInfo import dataInfo
from components.trainingPredictions import trainingPredictions
from components.trainDataSplit import trainDataSplit
from components.predDataProcess import predDataProcess
from components.finalPrediction import finalPrediction
from IPython.display import display

#loading the training data
training_data=pd.read_csv('data/training_dataset.csv')
display(training_data)

#training data information
dataInfo(training_data)

#training data preprocessing
data=preprocess(training_data)
X_train, X_test, y_train, y_test=trainDataSplit(data)

#Training data
display(X_train, "\n", y_train)

#model training and creation
model=model(X_train, y_train)

#predictions
trainingPredictions(X_test, y_test, model)

#loading the prediction data
prediction_data=pd.read_csv('data/final_prediction_set.csv')
display(prediction_data)

#prediction data preprocessing
data=preprocess(prediction_data)
X_pred=predDataProcess(data)

#final predictions
finalPrediction(X_pred, model)


