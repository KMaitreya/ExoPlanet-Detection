#preprocessing function

def preprocess(data):

    import pandas as pd
    import numpy as np

    data=data.copy()
    
    #dropping unused columns
    data=data.drop(['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition','koi_score'], axis=1)
    
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

    return data
    