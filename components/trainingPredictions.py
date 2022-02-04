#predictions from training

def trainingPredictions(X_test, y_test, model):

    from sklearn.metrics import classification_report
    import pandas as pd
    from IPython.display import display

    predictions=model.predict(X_test)
    predictions=pd.DataFrame(predictions, columns=['CONFIRMED', 'FALSE POSITIVE'])
    predictions=predictions.round()
    predictions=predictions.astype(int)
    display('\n\nActual set:\n', y_test)
    display('\n\nPredicted set:\n', predictions)

    loss, accuracy=model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: ", accuracy)

    print(classification_report(y_test, predictions))