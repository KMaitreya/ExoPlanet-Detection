#training data split

def trainDataSplit(data):

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split 

    #splitting data into features and target
    y=data[['CONFIRMED' , 'FALSE POSITIVE']]
    X=data.drop(['CONFIRMED', 'FALSE POSITIVE'], axis=1)

    #train test split
    X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
    
    #scaling the data
    scaler=StandardScaler()
    scaler.fit(X_train)
    scaler.fit(X_test)
    X_train=pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test=pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    #converting to numpy arrays
    X_train=X_train.to_numpy()
    X_test=X_test.to_numpy()

    #reshaping data for cnn
    X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test