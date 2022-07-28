
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

runPath = os.getcwd()

data_dir =runPath + '/data/'
mkdir(data_dir)

## train_dir
train_dir = data_dir + 'train/'
mkdir(train_dir)

## valid_dir
valid_dir = data_dir + 'valid/'
mkdir(valid_dir)

## doll_train
doll_dir = train_dir + 'doll/'
mkdir(doll_dir)

## doll_valid
doll_valid_dir = valid_dir + 'doll/'
mkdir(doll_valid_dir)

## blue_train
blue_dir = train_dir + 'blue/'
mkdir(blue_dir)

## blue_valid
blue_valid_dir = valid_dir + 'blue/'
mkdir(blue_valid_dir)

########################## Image Augmentation #################################
def img_transforms():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=50,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest',
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=30,
        seed=1,
        shuffle=True,
        class_mode='categorical'
    )


    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(150, 150),
        batch_size=30,
        seed=1,
        shuffle=False,
        class_mode="categorical"
    )
    return train_generator,valid_generator

train_generator,valid_generator = img_transforms()

############################# Complex Model ###################################
def cnn(width,height,depth,outputNum):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=16, 
                            kernel_size=3, 
                            padding='same', 
                            activation='relu', 
                            input_shape=[width, height, depth]), 

        tf.keras.layers.MaxPool2D(pool_size=3, strides=(3,3),padding='same'),

        tf.keras.layers.Conv2D(filters=64, 
                            kernel_size=3, 
                            padding='same', 
                            activation='relu'), 

        tf.keras.layers.MaxPool2D(pool_size=5, strides=(3,3),padding='same'),

        tf.keras.layers.Conv2D(filters=32, 
                            kernel_size=2, 
                            padding='same', 
                            activation='relu'), 

        tf.keras.layers.MaxPool2D(pool_size=3, strides=(2,2),padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(outputNum, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

model = cnn(150,150,3,2)



modelPath = './model'
mkdir(modelPath)

output_model_file = os.path.join(modelPath,"dollvssilver_weights.h5")

def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


TRAIN_STEP = 20


callbacks = [
            tf.keras.callbacks.TensorBoard(modelPath),
            tf.keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only=True,
                                            save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
        ]


history = model.fit(
        train_generator,
        epochs=TRAIN_STEP,
        validation_data = valid_generator,
        callbacks = callbacks
    )


plot_learning_curves(history, 'accuracy', TRAIN_STEP, 0, 1)
plot_learning_curves(history, 'loss', TRAIN_STEP, 0, 5)



############################## Simple Model ###################################
def cnn(width,height,depth,outputNum):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=16, 
                            kernel_size=5, 
                            padding='same', 
                            activation='relu', 
                            input_shape=[width, height, depth]), 

        tf.keras.layers.MaxPool2D(pool_size=5, strides=(5,5),padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(outputNum, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

model = cnn(150,150,3,2)

history = model.fit(
        train_generator,
        epochs=TRAIN_STEP,
        validation_data = valid_generator,
        callbacks = callbacks
    )

plot_learning_curves(history, 'accuracy', TRAIN_STEP, 0, 1)
plot_learning_curves(history, 'loss', TRAIN_STEP, 0, 5)
