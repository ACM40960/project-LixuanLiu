# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:49:18 2022

@author: Suiy
"""
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
silver_dir = train_dir + 'blue/'
mkdir(silver_dir)

## blue_valid
silver_valid_dir = valid_dir + 'blue/'
mkdir(silver_valid_dir)


def img_transforms():
    # 创建训练数据预处理模板
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255, # 所有数据集将乘以该数值（归一化，数据范围控制在0到1之间）
            rotation_range=40, # 随机旋转的范围
            width_shift_range=0.2, # 随机宽度偏移量
            height_shift_range=0.2, # 随机高度偏移量
            shear_range=0.2, # 随机错切变换
            zoom_range=0.2, # 随机缩放范围
            horizontal_flip=True, # 随机将一半图像水平翻转
            fill_mode='nearest', # 填充模式为最近点填充
    )

    # 从文件夹导入训练集
    train_generator = train_datagen.flow_from_directory(
        train_dir,# 训练数据文件夹
        target_size=(150, 150), # 处理后的图片大小128*128
        batch_size=30, # 每次训练导入多少张图片
        seed=123,# 随机数种子
        shuffle=True, # 随机打乱数据
        class_mode='categorical' # 返回2D的one-hot编码标签
    )

    # 创建校验数据预处理模板
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255, # 数据范围控制在0到1之间
    )

    # # 从文件夹导入校验集
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,# 校验数据文件夹
        target_size=(150, 150), # 处理后的图片大小128*128
        batch_size=30, # 每次训练导入多少张图片
        seed=123,# 随机数种子
        shuffle=False,# 随机打乱数据
        class_mode="categorical" # 返回2D的one-hot编码标签
    )
    return train_generator,valid_generator

train_generator,valid_generator = img_transforms()

############################# Model #####################################
def cnn(width,height,depth,outputNum):
    # 按顺序添加神经层
    # 神经网络的层数指的是有权重计算的层，池化层不算层数
    # 该模型卷积层6层 全连接层2层 共8层
    model = tf.keras.models.Sequential([
        # Conv2D 向两个维度进行卷积https://blog.csdn.net/qq_37774098/article/details/111997250
        # 卷积层 输入样本为宽：128 高：128 深度：3
        tf.keras.layers.Conv2D(filters=32, # 32个过滤器
                            kernel_size=5, # 核大小3x3
                            padding='same', # same：边缘用0填充 valid：边缘不填充
                            activation='relu', # 激活函数：relu
                            input_shape=[width, height, depth]), # 输入形状 宽：128 高：128 深度：3 不定义那就是默认输入的形状

        tf.keras.layers.MaxPool2D(pool_size=3, strides=(2,2),padding='same'),

        tf.keras.layers.Conv2D(filters=32, # 32个过滤器
                            kernel_size=3, # 核大小3x3
                            padding='same', # same：边缘用0填充 valid：边缘不填充
                            activation='relu'), # 激活函数：relu

        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2),padding='same'),
        # 输入层的数据压成一维的数据
        tf.keras.layers.Flatten(),

        # 全连接层
        tf.keras.layers.Dense(1024, activation='relu'),# 输出的维度大小128个特征点 激活函数：relu
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(outputNum, activation='softmax') # 输出猫狗类型 2类 猫或者狗 激活函数：softmax 用于逻辑回归
    ])

    model.compile(loss='categorical_crossentropy', # 损失函数 https://blog.csdn.net/qq_40661327/article/details/107034575
                  optimizer='adam',# 优化器 
                  metrics=['accuracy']) # 准确率

    # 打印模型每层的情况
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

# 定义训练步数
TRAIN_STEP = 10

# 设置回调模式
callbacks = [
            tf.keras.callbacks.TensorBoard(modelPath),
            tf.keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only=True,
                                            save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
        ]

# 开始训练
history = model.fit(
        train_generator,
        epochs=TRAIN_STEP,
        validation_data = valid_generator,
        callbacks = callbacks
    )

# 显示训练曲线
plot_learning_curves(history, 'accuracy', TRAIN_STEP, 0, 1)
plot_learning_curves(history, 'loss', TRAIN_STEP, 0, 5)











