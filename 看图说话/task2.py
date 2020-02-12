from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras

def load_img_as_np_array(filepath, target_size):
    '''
    从给定文件加载图像，转换图像大小为给定的target_size，返回keras支持的浮点数numpy数组
    :param path:图像文件路径
    :param target_size:元组(图像高度，图像宽度)
    :return:numpy数组
    '''
    img = pil_image.open(filepath)
    img.resize(target_size, pil_image.NEAREST)
    return np.asarray(img, dtype=K.floatx())

def preprocess_input(x):
    '''
    预处理图像用于网络输入，将图像由RGB转为BGR
    将图像的每一个图像通道减去其均值
    均值BGR三个通道的均值分别为103.939,116.779,123.68
    :param x:数组，4维
    :return:
    '''
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

def load_vgg16_model():
    '''
    将两个文件导入
    https://pan.baidu.com/s/13WQBRb4sr3umP7xbUCxmCg  ycb5
    https://pan.baidu.com/s/1yF8wybHuzGoTzwSkqTPzzQ  ub75
    :return: model
    '''
    json_file = open("vgg16_exported.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("vgg16_exported.h5")
    return model

def extract_fertures(directory):
    '''
    提取给定文件夹中所有图像的特征，将提取的特征保存在文件features.pkl中
    提取的特征保存在一个dict中，key为文件名（不带.jpg），value为特征值[np.array]
    :param directory:包含jpg文件的文件夹
    :return:None
    '''
    # 导入model
    model = load_vgg16_model()
    # 去除模型最后一层
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    features =dict()  # 定义数据字典
    for fn in listdir(directory):
        fn = directory + '/' + fn
        arr = load_img_as_np_array(fn, target_size=(224, 224))

        # 改变数组的形态，增加一个维度（批处理输入的维度）
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        # 预处理图像作为VGG模型的输入
        arr=preprocess_input(arr)
        # 计算特征
        feature = model.predict(arr, verbose=0)

        id = fn  # 去掉文件后缀和directory
        features[id] = feature

if __name__ =='__main__':
    '''
    提取Flicker8k数据集中所有图像的特征，保存在一个文件中，下载链接：
    https://pan.baidu.com/s/1bQcQAz0pxPix9q9kCoZ1aw  6gpd
    '''
    directory = '.\Flicker8k_Dataset'
    features = extract_fertures(directory)
    print('提取特征的文件个数：%d' % len(features))
    print(keras.backend.image_data_format())
    # 保存特征到文件
    dump(features, open('features.pkl', 'wb'))