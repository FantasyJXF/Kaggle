# !/usr/bin/env python
# -*- coding=utf-8 -*-

# 数据增强
import os
import PreProcess
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt


# 数据增强效果测试
"""
datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')

fnames = [os.path.join(PreProcess.train_cats_dir, fname) 
          for fname in os.listdir(PreProcess.train_cats_dir)]
# 选择一张图像
img_path = fnames[3]
#读取图像并调整大小
img = image.load_img(img_path, target_size = (150, 150))
#将其转换成形状(150, 150, 3)的Numpy数组
x = image.img_to_array(img)
#将其形状改变为(1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size = 1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i = i + 1
    if i % 4 == 0:
        break  
plt.show()
"""

def create_dataflow():
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        PreProcess.train_dir,
        target_size = (150, 150),
        batch_size = 32,
        class_mode = 'binary')

    validation_generator = test_datagen.flow_from_directory(
        PreProcess.validation_dir,
        target_size = (150, 150),
        batch_size = 32,
        class_mode = 'binary')

    test_generator = test_datagen.flow_from_directory(
        PreProcess.test_dir,
        target_size = (150, 150),
        batch_size = 32,
        class_mode = 'binary')

    return train_generator, validation_generator, test_generator

if __name__ == "__main__":
    create_dataflow()