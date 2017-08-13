from __future__ import division

from keras.models import Model
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import add, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import load_model

import numpy as np
from PIL import Image
import glob
import six
import random
import pandas as pd

np.random.seed(2017)
random.seed(2017)


conf = dict()

#设置训练的数量和测试的数量
conf['train_valid_fraction'] = 0.8

#设置每次载入的数量
conf['batch_size'] = 20

#设置训练的次数
conf['epochs'] = 30

#当训练epochs次后，没有提升停止训练
conf['patience'] = 5

#设置CNN图像的大小

conf['image_shape'] = (1024, 1024)


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _bn_relu(input):
    '''建立BN -> relu block'''
    bn = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation('relu')(bn)

def _conv_bn_relu(**conv_params):
    '''建立 conv -> BN -> relu block'''
    filters = conv_params["filters"]
    row = conv_params["row"]
    col = conv_params["col"]
    strides = conv_params.setdefault("strides", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    regularizer = conv_params.setdefault("regularizer", l2(1.e-4))

    def f(input):
        conv = Convolution2D(filters=filters, kernel_size=(row, col), strides=strides,
                             kernel_initializer=init,
                             padding=padding, kernel_regularizer=regularizer)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv(**conv_params):
    '''建立BN -> relu -> conv block'''

    filters = conv_params["filters"]
    row = conv_params["row"]
    col = conv_params["col"]
    strides = conv_params.setdefault("strides", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    regularizer = conv_params.setdefault("regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Convolution2D(filters=filters, kernel_size=(row, col), strides=strides, kernel_initializer=init,
                             padding=padding, kernel_regularizer=regularizer)(activation)

    return f


def _shortcut(input, residual):
    '''添加shortcut，合并input, residual 使用add'''
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(filters=residual_shape[CHANNEL_AXIS],
                                 kernel_size=(1, 1),
                                 strides=(stride_width, stride_height),
                                 kernel_initializer="he_normal", padding="valid",
                                 kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer = False):
    '''建立residual block'''
    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = block_function(filters=filters, strides=strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, strides=(1, 1), is_first_block_of_first_layer=False):
    """3*3的kernel_size, 小于34层使用
    """
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Convolution2D(filters = filters,
                                 kernel_size=(3,3),
                                 strides=strides,
                                 kernel_initializer="he_normal", padding="same",
                                 kernel_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, row=3, col=3,
                                  strides=strides)(input)

        residual = _bn_relu_conv(filters=filters, row=3, col=3)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, strides=(1, 1), is_first_block_of_first_layer=False):
    """大于34层使用
    """
    def f(input):

        if is_first_block_of_first_layer:
            conv_1_1 = Convolution2D(filters=filters,
                                 kernel_size=(1,1),
                                 strides=strides,
                                 kernel_initializer="he_normal", padding="same",
                                 kernel_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, row=1, col=1,
                                     strides=strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, row=3, col=3)(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, row=1, col=1)(conv_3_3)
        return _shortcut(input, residual)

    return f


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """建立模型

        Args:
            input_shape: 输入的类型 (channels, rows, cols)
            num_outputs: 输出的类型
            block_fn: 使用 `basic_block` 或者 `bottleneck`.
                使用basic_block 当层数 < 50
            repetitions: 模块

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (channels, rows, cols)")


        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])


        block_fn = _get_block(block_fn)

        #输入
        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, row=7, col=7, strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        #输出
        block = _bn_relu(block)

        block_norm = BatchNormalization(axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)


        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal", activation="softmax")(flatten1)
        #dense = Dense(output_dim=num_outputs, W_regularizer=l2(0.01), init="he_normal", activation="linear")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_test(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1, 1, 1])

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


def batch_generator_train(files, batch_size):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    while True:
        batch_files = files[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            image = Image.open(f)
            image = image.resize(conf['image_shape'])
            image = np.array(image, dtype='float32')
            #print(image.shape)

            cancer_type = f[16:17] # 输出的类型
            #print(f[16:17])
            #print(f)
            if cancer_type == '1':
                mask = [1, 0, 0]
            elif cancer_type == '2':
                mask = [0, 1, 0]
            else:
                mask = [0, 0, 1]
            #print(mask)

            image_list.append(image)
            mask_list.append(mask)
        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        yield (image_list, mask_list)

        if counter == number_of_batches:
            random.shuffle(files)
            counter = 0



def train():
    filepaths = []
    filepaths.append('data/train/Type_1/')
    filepaths.append('data/train/Type_2/')
    filepaths.append('data/train/Type_3/')

    allFiles = []

    for i, filepath in enumerate(filepaths):
        files = glob.glob(filepath + '*.jpg')
        allFiles = allFiles + files

    split_point = int(round(conf['train_valid_fraction'] * len(allFiles)))

    random.shuffle(allFiles)

    train_list = allFiles[:split_point]
    valid_list = allFiles[split_point:]
    print('Train patients: {}'.format(len(train_list)))
    print('Valid patients: {}'.format(len(valid_list)))



    print('Create and compile model...')

    classes = 3
    img_rows, img_cols = conf['image_shape'][1], conf['image_shape'][0]
    img_channels = 3

    model = ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='hinge',optimizer='adadelta',metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=0),
        ModelCheckpoint('cervical_best.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
        TensorBoard(log_dir='logs')
    ]

    print('Fit model...')
    model.fit_generator(generator=batch_generator_train(train_list, conf['batch_size']),
                              epochs=conf['epochs'],
                              steps_per_epoch=int(len(allFiles) / conf['batch_size']),
                              validation_data=batch_generator_train(valid_list, conf['batch_size']),
                              validation_steps=3,
                              verbose=1,
                              callbacks=callbacks)
    print('End fitting')
    image_list = []
    f = 'data/train/Type_1/0.jpg'
    image = Image.open(f)
    image = image.resize(conf['image_shape'])
    image = np.array(image, dtype='float32')
    image_list.append(image)

    image_list = np.array(image_list)

    predictions = model.predict(image_list, verbose=1, batch_size=1)
    print(predictions)


def predict():
    model = load_model('cervical_best.hdf5')

    sample_subm = pd.read_csv("data/sample_submission.csv")
    ids = sample_subm['image_name'].values

    for id in ids:
        print('Predict for image {}'.format(id))
        files = glob.glob("data/test/" + id)
        image_list = []
        for f in files:
            image = Image.open(f)
            image = image.resize(conf['image_shape'])
            image = np.array(image, dtype='float32')
            image_list.append(image)

        image_list = np.array(image_list)

        predictions = model.predict(image_list, verbose=1, batch_size=1)


        sample_subm.loc[sample_subm['image_name'] == id, 'Type_1'] = predictions[0, 0]
        sample_subm.loc[sample_subm['image_name'] == id, 'Type_2'] = predictions[0, 1]
        sample_subm.loc[sample_subm['image_name'] == id, 'Type_3'] = predictions[0, 2]

    sample_subm.to_csv("subm.csv", index=False)


if __name__ == "__main__":
    train()