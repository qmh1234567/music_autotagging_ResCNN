#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2019/06/21 16:14:42
@Author  :   Four0Eight
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPool2D, Dropout,Activation
from keras.layers import GRU, ELU, Reshape, Dense,Lambda,Add
from keras.regularizers import l2

from keras import Model
import constants as c


def cnn_block(model, num_filters, pool_size, layer_id):
    model.add(Conv2D(num_filters, 3, padding='same', name=f'conv{layer_id}'))
    model.add(BatchNormalization(axis=-1, name=f'bn{layer_id}'))
    model.add(ELU(name=f'elu{layer_id}'))
    model.add(MaxPool2D(pool_size, name=f'pool{layer_id}'))
    model.add(Dropout(0.1, name=f'dropout{layer_id}'))

    return model

def music_crnn(input_shape, num_class):
    model = Sequential()

    model.add(ZeroPadding2D((0, 37), input_shape=input_shape))
    model = cnn_block(model, 64, (3,3), 1)
    model = cnn_block(model, 128, (2,2), 2)
    model = cnn_block(model, 128, (4,4), 3)
    model = cnn_block(model, 128, (4,4), 4)

    model.add(Reshape((15, 128)))

    model.add(GRU(32, return_sequences=True, name='gru1'))
    model.add(GRU(32, return_sequences=False, name='gru2'))
    model.add(Dropout(0.3, name=f'dropout5'))

    model.add(Dense(num_class, activation='sigmoid', name='output'))

    model.summary()

    return model

def res_conv_block(x,filters,strides,name):
    filter1,filter2,filter3 = filters
    shortcut = Conv2D(filter3,(1,1),strides=strides,use_bias=True,name=f'{name}_scut_conv',
    kernel_regularizer = l2(c.WEIGHT_DECAY),kernel_initializer='glorot_uniform')(x)
    shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)

    # block a 
    x = Conv2D(filter1,(1,1),strides=strides,use_bias=True,name=f'{name}_conva',
    kernel_regularizer=l2(c.WEIGHT_DECAY),kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(name=f'{name}_bna')(x)
    x = Activation('relu',name=f'{name}_relua')(x)
    # block b
    x = Conv2D(filter2,(3,3),padding='same',use_bias=True,name=f'{name}_convb',
    kernel_regularizer=l2(c.WEIGHT_DECAY),kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(name=f'{name}_bnb')(x)
    x = Activation('relu',name=f'{name}_relub')(x)
    # block c
    x = Conv2D(filter3,(1,1),use_bias=True,name=f'{name}_convc',
    kernel_regularizer=l2(c.WEIGHT_DECAY),kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization(name=f'{name}_bnc')(x)

    x = Add(name=f'{name}_scut')([shortcut,x])
    x = Activation('relu',name=f'{name}_relu1')(x)
    return x


def ResCNN(input_shape,num_class):

    x_in = Input(input_shape,name='input')

    x = ZeroPadding2D((2,37),name='zero-padding')(x_in)

    x = Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv1')(x_in)
    x = BatchNormalization(name='norm1')(x)
    x = Activation('relu',name='relu1')(x)


    x = Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv2')(x)
    x = BatchNormalization(name='norm2')(x)
    x = Activation('relu',name='relu2')(x)

    x = MaxPool2D((3,3),strides=(2,2),padding='same',name='pool2')(x)

    x = res_conv_block(x,[64,64,256],strides=(2,2),name='block1')
    x = res_conv_block(x,[64,64,256],strides=(2,2),name='block2')
    x = res_conv_block(x,[64,64,256],strides=(2,2),name='block3')

    x = res_conv_block(x,[128,128,512],strides=(2,2),name='block4')
    x = res_conv_block(x,[128,128,512],strides=(2,2),name='block5')
    x = res_conv_block(x,[128,128,512],strides=(2,2),name='block6')

    # 减少维数
    x = Lambda(lambda y: K.mean(y,axis=[1,2]),name='avgpool')(x)
    # the final two fcs
    x = Dense(x.shape[-1].value,kernel_initializer='glorot_uniform',name='final_fc')(x)
    x = BatchNormalization(name='final_norm')(x)
    x = Activation('relu',name='final_relu')(x)

    x = Dropout(0.2,name='final_drop')(x)
    x = Dense(num_class,kernel_initializer='glorot_uniform',name='logit')(x)
    x = Activation('sigmoid',name='pred')(x)
    return Model(inputs=[x_in],outputs=[x],name='ResCNN')

if __name__ == "__main__":
    # model = music_crnn((96,1366,1),len(c.TAGS))
    model = ResCNN((96,1366,1),len(c.TAGS))
    print(model.summary())
    