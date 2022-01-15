def model(X_train, y_train):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    #creating the model
    model=Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=3, activation='sigmoid'))
    #creating 3 output nodes for 3 classes(CONFIRMED, CANDIDATE, FALSE POSITIVE)

    #compiling the model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

    #training the model
    model.fit(X_train, y_train, epochs=40, batch_size=10, verbose=1)

    return model