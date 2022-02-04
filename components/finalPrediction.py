#final predictions on candidate  data

def finalPrediction(X_pred, model):
    
    import pandas as pd
    from IPython.display import display
    from tensorflow.keras.models import Sequential

    predictions=model.predict(X_pred)
    predictions=pd.DataFrame(predictions, columns=['CONFIRMED', 'FALSE POSITIVE'])
    predictions=predictions.round()
    predictions=predictions.astype(int)
    display('\n\nPredicted set:\n\n', predictions)