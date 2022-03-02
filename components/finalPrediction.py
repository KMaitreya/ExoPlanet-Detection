#final predictions on candidate  data

def finalPrediction(X_pred, model):
    
    import pandas as pd

    predictions=model.predict(X_pred)
    predictions=pd.DataFrame(predictions, columns=['CONFIRMED', 'FALSE POSITIVE'])
    predictions=predictions.round()
    predictions=predictions.astype(int)
    
    return predictions