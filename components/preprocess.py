#preprocessing function

def preprocess(data):

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    data=data.copy()
    
    #dropping unused columns
    data=data.drop(['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition','koi_score'], axis=1)
    
    #dropping columns which have all missing values
    data=data.drop(['koi_teq_err1', 'koi_teq_err2'], axis=1)
    
    #filling missing values
    data['koi_tce_delivname']=data['koi_tce_delivname'].fillna(data['koi_tce_delivname'].mode()[0])
    for col in data.columns[data.isna().sum()>0]:
        data[col]=data[col].fillna(data[col].mean())
    
    #creating separate columns for koi_tce_delivname
    dummies=pd.get_dummies(data['koi_tce_delivname'], prefix='delivname')
    data=pd.concat([data, dummies], axis=1)
    data=data.drop(['koi_tce_delivname'], axis=1)

    #splitting disposition into 3 columns(CONFIRMED, CANDIDATE, FALSE POSITIVE)
    dummies=pd.get_dummies(data['koi_disposition'])
    data=pd.concat([data, dummies], axis=1)
    data=data.drop(['koi_disposition'], axis=1)
    
    #splitting data into features and target
    y=data[['CONFIRMED' , 'CANDIDATE', 'FALSE POSITIVE']]
    X=data.drop(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], axis=1)
    
    #train test split
    X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.8, random_state=1, shuffle=True)
    
    #scaling the data
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test=pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test
    