import numpy as np
import pandas as pd
from components.preprocess import preprocess
from components.model import model
from components.dataInfo import dataInfo
from IPython.display import display

#loading the data
data=pd.read_csv('dataset/cumulative.csv')
display(data)

#data information
dataInfo(data)

#running the preprocess function
X_train, X_test, y_train, y_test=preprocess(data)

#Training data
display(X_train, "\n", y_train)

#model training and creation
model=model(X_train, y_train)

#predictions
predictions=model.predict(X_test)
predictions=pd.DataFrame(predictions, index=X_test.index, columns=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
predictions=predictions.round(0)
display('\nn\ Actual set:\n', y_test)
display('Predicted set:\n', predictions)

loss, accuracy=model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: ", accuracy)