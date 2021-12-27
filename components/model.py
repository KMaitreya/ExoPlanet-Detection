def model(X_train, y_train):
    
    from sklearn.neural_network import MLPClassifier
    
    model=MLPClassifier(hidden_layer_sizes=(150))
    model.fit(X_train, y_train)

    return model