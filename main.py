import numpy as np
import pandas as pd
from components.preprocess import preprocess
from components.model import model
from components.dataInfo import dataInfo
from sklearn.metrics import classification_report

#loading the data
data=pd.read_csv('dataset/cumulative.csv')
data

#data information
dataInfo(data)

#running the preprocess function
X_train, X_test, y_train, y_test=preprocess(data)

#Training data
print(X_train, "\n", y_train)

#model training and creation
model=model(X_train, y_train)

#predictions
y_pred=model.predict(X_test)

#classification report
print(classification_report(y_test, y_pred))