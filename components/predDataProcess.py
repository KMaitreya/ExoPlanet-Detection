#training data split

def predDataProcess(data):

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    #splitting data into features and target
    X_pred=data.drop(['CANDIDATE'], axis=1)
    
    #scaling the data
    scaler=StandardScaler()
    scaler.fit(X_pred)
    X_pred=pd.DataFrame(scaler.transform(X_pred), index=X_pred.index, columns=X_pred.columns)

    #converting to numpy arrays
    X_pred=X_pred.to_numpy()

    #reshaping data for cnn
    X_pred=np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

    return X_pred