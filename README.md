# <center>_Image Recognition Based on CNN_</center>

<center><img src="https://img.shields.io/badge/python-3.8.8-blue.svg"/> <img src="https://img.shields.io/badge/tensorflow-2.3.0-green.svg"/> </center>

## _Table of Content_

- [<center>_Image Recognition Based on CNN_</center>](#centerimage-recognition-based-on-cnncenter)
  - [_Table of Content_](#table-of-content)
  - [_Basic Overview_](#basic-overview)
  - [_Tools and Preparation_](#tools-and-preparation)
  - [Preparation](#preparation)
    - [Making the dirctionary of first level](#making-the-dirctionary-of-first-level)
    - [Making the dirctionary of second level](#making-the-dirctionary-of-second-level)
  - [Download data here (Very important):](#download-data-here-very-important)
  - [Image Augmentation](#image-augmentation)
  - [Modeling Training](#modeling-training)
    - [Complex Model](#complex-model)
    - [Simple Model](#simple-model)


## _Basic Overview_
The overall idea of this project is to build up a Convolutional Neural Network which can automatically tell to which breed of cat belong based on their coat, shape of face and some other elements once the cats' pictures are provided.

[(Back to top)](#table-of-content)

## _Tools and Preparation_

- Python 3.8.8  

    You can choose any editor you want, a few options are provide here:
    - Spyder (Anaconda)
    - Jupyter Notebook (Anaconda)

    Both of the editor can be find here: [Anaconda Download](https://www.anaconda.com/products/distribution) .

<br />

- Tensorflow 2.3.0  

    To support Python 3.8, TensorFlow 2.2 or higher is required
    
    Firstly we seperate a new environment for tensorflow, so that the original environment won't be influenced
    
    the name of the new environment can be named as whatever you like
    
    ![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/python_environment.png)
    
    
    then open Anaconda Prompt from the start manue
    
    ![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/Tools_Anaconda.png)
    
        
    ```bash
    # choose the new environment (need to change the name, the name of my environment is tensorflow)
    activate tensorflow
    # Current stable release for CPU and GPU
    pip install tensorflow
    ```
    
<br />

- Python Packages
To run the code, a few packages are needed. This can be done by from Anaconda or Anaconda Prompt. 

Here I will show the first way.

Go to the environment and choose you new environment created in the last step.

choose the `Not Installed` and type in the name of the package you want in the input box.

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/packages.png)

The packages that we need to install are:

  - pandas
  - scipy
  - matplotlib

And the version of package pillow need to be change, go the `installed` and type in pillow, choose the version 9.0.1

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/pillow_version.png)

The editor I choose is spyder. Choose the the tensorflow environment and install spyder

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/spyder.png)

Then we can import packages in Python

  ```python
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
  ```

[(Back to top)](#table-of-content)

## Preparation

- `The whole program need to be run under tensorflow environment.`
- First of all, data for this model need to be placed into right palce.

### Making the dirctionary of first level

```python
    def mkdir(path):
    if not os.path.exists(path):
    os.makedirs(path)
    
    ## get main directionary
    runPath = os.getcwd()
    
    data_dir =runPath + '/data/'
    mkdir(data_dir)

    ## train_dir
    train_dir = data_dir + 'train/'
    mkdir(train_dir)

    ## valid_dir
    valid_dir = data_dir + 'valid/'
    mkdir(valid_dir)
```

Creating two folders `train, valid` under default dictionary:
- train: place training data, contain two folders, folder "blue" to store the images of the Britain Blue, folder "doll" to store the images
- valid: place validation data

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/mkdir_train_valid.png)

### Making the dirctionary of second level

```python
    ## doll_train
    doll_dir = train_dir + 'doll/'
    mkdir(doll_dir)
    
    ## blue_train
    blue_dir = train_dir + 'blue/'
    mkdir(blue_dir)
    
```

Under train folder, we store images in two places:
- blue: for images of British Blue Cat
- doll: for images of Ragdoll

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/mkdir_train.png)

<br />

```python
    ## doll_valid
    doll_valid_dir = valid_dir + 'doll/'
    mkdir(doll_valid_dir)

    ## blue_valid
    blue_valid_dir = valid_dir + 'blue/'
    mkdir(blue_valid_dir)
```
    
Same as above:
- blue: for images of British Blue Cat
- doll: for images of Ragdoll   

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/mkdir_valid.png)

<br />

## Download data here (Very important):

After creating dirctionary, now we need data for our model !!

Data have been uploaded to my Google drive, and I will also submit another copy through `Brightsapce`.

Here is the Link to Google Drive: [CNN-Data](https://drive.google.com/file/d/1eUgZoiIiYIRRCsoZt_K7Y2kgBxRmGrlS/view?usp=sharing)

[(Back to top)](#table-of-content)

## Image Augmentation

The purpose of Image Augmentation is makeing the sample data more comprehensive

```python
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
```
If all datasets are placed correctly, the result of the Image Augnmentation should be like:

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/Image_augmentation.png)

[(Back to top)](#table-of-content)

## Modeling Training

We have two models, one is a complex model, another one has simpler structure

### Complex Model

```python
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
```
The output will be the summary the model

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/comple_model.png)

Deciding the parameters of the model is the first step, then the second step is set the callback rules:

```python
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
```

The final step is training the model and 

```python
history = model.fit(
        train_generator,
        epochs=TRAIN_STEP,
        validation_data = valid_generator,
        callbacks = callbacks
    )
```

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/process_normal.png)

plot the accuracy curve as well as the loss curve

```python
plot_learning_curves(history, 'accuracy', TRAIN_STEP, 0, 1)
plot_learning_curves(history, 'loss', TRAIN_STEP, 0, 5)
```

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/acc_normal.png)

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/loss_normal.png)

[(Back to top)](#table-of-content)

### Simple Model

```python
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
```

![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/simple_model.png)

The callback rules and plot code for the simple model is just same as above, the result is shown in the report document

[(Back to top)](#table-of-content)
