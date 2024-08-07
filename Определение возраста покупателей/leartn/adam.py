from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
 
 
def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
    return features_train, target_train
 
 
def create_model(input_shape):
 
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(4,4), padding='same', strides=1, activation='relu', input_shape=(28, 28, 1)))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None, padding='valid', ))
    model.add(Conv2D(filters=16, kernel_size=(4,4), padding='valid', strides=1, activation='relu', input_shape=(10, 10, 16)))
    model.add(AvgPool2D(pool_size=(2, 2), strides=None, padding='valid', ))
    model.add(Flatten())
    model.add(Dense(units=120, activation="relu", input_dim =(None, 120)))
    model.add(Dense(units=84, activation="relu", input_dim =(None, 84)))
    model.add(Dense(units=10, activation="softmax", input_dim =(None, 10)))
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
              metrics=['acc']) 
 
 
    return model
 
 
def train_model(model, train_data, test_data, batch_size=32, epochs=5,
               steps_per_epoch=None, validation_steps=None):
 
    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)
 
    return model