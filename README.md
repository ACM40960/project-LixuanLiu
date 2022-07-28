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

[!image](https://github.com/ACM40960/project-LixuanLiu/blob/main/packages.png)

The packages that we need to install are:

  - pandas
  - scipy
  - matplotlib

And the version of package pillow need to be change, go the `installed` and type in pillow, choose the version 9.0.1



And the we can import packages in Python

  ```python
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
  ```

[(Back to top)](#table-of-content)

## Preparation

- `The whole program need to be run under tensorflow`
- First of all some data need to be placed into right palce

### Making the dirctionary of first level

```python
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
```

Creating two folders `train, valid` under default dictionary:
- train: place training data
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
