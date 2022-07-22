# <center>_Image Recognition Based on CNN_</center>

<center><img src="https://img.shields.io/badge/python-3.8.8-blue.svg"/> <img src="https://img.shields.io/badge/tensorflow-2.3.0-green.svg"/> </center>

## _Table of Content_
- [<center>_Image Recognition Based on CNN_</center>](#centerimage-recognition-based-on-cnncenter)
  - [_Table of Content_](#table-of-content)
  - [_Basic Overview_](#basic-overview)
  - [_Tools and Preparation_](#tools-and-preparation)


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
    
    Open Anaconda Prompt
    
    ![image](https://github.com/ACM40960/project-LixuanLiu/blob/main/Tools_Anaconda.png)
    
    type in
    
    ```bash
    # Requires the latest pip
    $ pip install --upgrade pip
    # Current stable release for CPU and GPU
    $ pip install tensorflow
    ```

<br />

- Python Packages
  ```python
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
  ```

[(Back to top)](#table-of-content)


