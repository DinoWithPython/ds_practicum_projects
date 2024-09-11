from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
 
def load_train(path):
 
    train_datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1./255,
    #horizontal_flip=True
    #vertical_flip=True,
    #rotation_range=90, 
    #width_shift_range=0.2, 
    #height_shift_range=0.2    
    )
 
    train_datagen_flow = train_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    seed=12345)
 
    return train_datagen_flow
 
def load_valid(path):
 
    validation_datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1./255)
 
    val_datagen_flow = validation_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='validation',
    seed=12345) 
 
    return val_datagen_flow
 
def create_model(input_shape,lr=0.001):
    model = Sequential()
    optimizer = Adam(lr)
    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                     activation="relu", input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=12, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model
 
def train_model(model, train_data, test_data, batch_size=None, epochs=4,
                steps_per_epoch=None, validation_steps=None):
 
    model.fit(train_data,
            validation_data=test_data,
            batch_size = batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2, epochs=epochs)
 
    return model      
 
 
if __name__ == "__main__":    
 
    train_datagen_flow = load_train('/datasets/fruits_small/')
    valid_datagen_flow = load_valid('/datasets/fruits_small/')  
    model = create_model(input_shape=(150, 150, 3),lr=0.0001) 
    model = train_model(model, train_datagen_flow, valid_datagen_flow, epochs=4)    
    #print("Model accuracy:")
    loss, acc = model.evaluate(valid_datagen_flow[0], valid_datagen_flow[1], verbose=2)     
    print("Model accuracy: {:5.2f}%".format(100 * acc))