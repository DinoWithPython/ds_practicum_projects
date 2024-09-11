import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
 
 
def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(features_train.shape[0], 28 * 28) / 255.
    return features_train, target_train
 
 
def create_model(input_shape):
    model = Sequential()
    
    model.add(Dense(units=500, input_shape=input_shape, activation="relu"))
    model.add(Dense(units=300, activation="relu"))
    model.add(Dense(units=10, activation='softmax'))
    return model
 
 
def train_model(model, train_data, test_data, batch_size=48, epochs=50,
               steps_per_epoch=None, validation_steps=None):
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
              metrics=['acc']) 
    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
 
    return model