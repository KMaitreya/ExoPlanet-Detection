def model(X_train, y_train):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten
    
    #creating the model
    model=Sequential()
    model.add(Conv1D(8, 1, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(16, 1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    #compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

    #training the model
    cnn_model=model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

    return model