def visualize(cnn_model):
    
    import matplotlib.pyplot as plt
    
    # summarize history for accuracy
    plt.plot(cnn_model.history['accuracy'])
    plt.plot(cnn_model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.show()

    # summarize history for loss
    plt.plot(cnn_model.history['loss'])
    plt.plot(cnn_model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()