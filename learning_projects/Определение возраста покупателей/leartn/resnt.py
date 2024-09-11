from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
 
 
def load_train(path):
 
    train_datagen = ImageDataGenerator(rescale=1/255., horizontal_flip=True, vertical_flip=True)
    #, width_shift_range=0.2, height_shift_range=0.2, rotation_range=90 validation_split=0.25, 
    #validation_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255.)    
    #train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
 
    train_datagen_flow = train_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
   # subset='training',
    seed=12345)
 
    #val_datagen_flow = validation_datagen.flow_from_directory(
    #'/datasets/fruits_small/',
    #target_size=(150, 150),
    #batch_size=16,
    #class_mode='sparse',
    #subset='validation',
    #seed=12345)
 
    #features, target = next(train_datagen_flow)
 
    return train_datagen_flow
 
 
def create_model(input_shape):
 
 
    backbone = ResNet50(input_shape=(150, 150, 3),
   #                 weights='imagenet', 
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                    include_top=False)
 
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
 
    #model.add(Flatten())
 
    #model = Sequential()
    #model.add(Dense(100, input_shape=input_shape, activation='relu'))
    #model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same',
    #             activation='relu', input_shape=input_shape))
    #model.add(AvgPool2D(pool_size=(2, 2)))
    #model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
    #             activation='relu'))
    #model.add(AvgPool2D(pool_size=(2, 2)))
    #model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    #model.add(Dense(units=60, activation='relu'))
    model.add(Dense(units=42, activation='relu'))
    #model.add(Dense(units=22, activation='relu'))
    #model.add(Dense(units=12, activation='relu'))
 
    model.add(Dense(units=12, activation='softmax'))
    optimizer = Adam(lr=0.0001) #HIER ANPASSRN
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
 
    return model
 
 
def train_model(model, train_data, test_data, batch_size=None, epochs=3,
              steps_per_epoch=None, validation_steps=None):
 
#    if steps_per_epoch is None:        
#        steps_per_epoch = len(train_datagen_flow) 
#    if validation_steps is None:
#        validation_steps = len(test_data)
 
    model.fit(train_data, validation_data=test_data, 
              batch_size=batch_size, epochs=epochs, 
              steps_per_epoch=steps_per_epoch, 
              validation_steps=validation_steps, 
              verbose=2, shuffle=True)
 
    return model 