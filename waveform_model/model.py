from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply, GlobalMaxPool1D,
                                     Dense, Dropout, Activation, Reshape, Concatenate, Add, Input)
from tensorflow.keras.regularizers import l2
from lib.initialization import taejun_uniform
from tensorflow.keras.layers import Conv2D,Reshape,Lambda,MaxPool2D
import tensorflow.keras.backend as K

# import keras.backend as K
def squeeze_excitation(x, amplifying_ratio, name):
  num_features = x.shape[-1].value   
  x = GlobalAvgPool1D(name=f'{name}_squeeze')(x)
  x = Reshape((1, num_features), name=f'{name}_reshape')(x)
  x = Dense(num_features * amplifying_ratio, activation='relu',
            kernel_initializer='glorot_uniform', name=f'{name}_ex0')(x)
  x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform', name=f'{name}_ex1')(x)
  return x


def basic_block(x, num_features, cfg, name):
  """Block for basic models."""
  x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(), name=f'{name}_conv')(x)
  x = BatchNormalization(name=f'{name}_norm')(x)
  x = Activation('relu', name=f'{name}_relu')(x)
  x = MaxPool1D(pool_size=3, name=f'{name}_pool')(x)
  return x


def se_block(x, num_features, cfg, name):
  """Block for SE models."""
  x = basic_block(x, num_features, cfg, name)
  #  返回逐元素乘积的张量
  x = Multiply(name=f'{name}_scale')([x, squeeze_excitation(x, cfg.amplifying_ratio, name)])
  return x


def rese_block(x, num_features, cfg, name):
  """Block for Res-N & ReSE-N models."""
  if num_features != x.shape[-1].value:
    shortcut = Conv1D(num_features, kernel_size=1, padding='same', use_bias=True, name=f'{name}_scut_conv',
                      kernel_regularizer=l2(cfg.weight_decay), kernel_initializer='glorot_uniform')(x)
    shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)
  else:
    shortcut = x

  for i in range(cfg.num_convs):
    if i > 0:
      x = Activation('relu', name=f'{name}_relu{i-1}')(x)
      x = Dropout(0.2, name=f'{name}_drop{i-1}')(x)
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
               kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(), name=f'{name}_conv{i}')(x)
    x = BatchNormalization(name=f'{name}_norm{i}')(x)

  # Add SE if it is ReSE block.
  if cfg.amplifying_ratio:
    x = Multiply(name=f'{name}_scale')([x, squeeze_excitation(x, cfg.amplifying_ratio, name)])

  x = Add(name=f'{name}_scut')([shortcut, x])
  x = Activation('relu', name=f'{name}_relu1')(x)
  x = MaxPool1D(pool_size=3, name=f'{name}_pool')(x)
  return x


def SampleCNN(inputshape,cfg):
  """Build a SampleCNN model."""
  # Variable-length input for feature visualization.
  x_in = Input(inputshape, name='input')

  num_features = cfg.init_features
  x = Conv1D(num_features, kernel_size=3, strides=3, padding='same', use_bias=True,
             kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(scale=1.), name='conv0')(x_in)
  x = BatchNormalization(name='norm0')(x)
  x = Activation('relu', name='relu0')(x)

  # Stack convolutional blocks.
  layer_outputs = []
  for i in range(cfg.num_blocks):
    num_features *= 2 if (i == 2 or i == (cfg.num_blocks - 1)) else 1
    x = cfg.block_fn(x, num_features, cfg, f'block{i}')
    layer_outputs.append(x)

  if cfg.multi:  # Use multi-level feature aggregation or not.
    x = Concatenate(name='multi')([GlobalMaxPool1D(name=f'final_pool{i}')(output)
                                   for i, output in enumerate(layer_outputs[-3:])])
  else:
    x = GlobalMaxPool1D(name='final_pool')(x)

  # The final two FCs.
  x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform', name='final_fc')(x)
  x = BatchNormalization(name='final_norm')(x)
  x = Activation('relu', name='final_relu')(x)
  if cfg.dropout > 0.:
    x = Dropout(cfg.dropout, name='final_drop')(x)
  x = Dense(cfg.num_classes, kernel_initializer='glorot_uniform', name='logit')(x)
  x = Activation(cfg.activation, name='pred')(x)

  return Model(inputs=[x_in], outputs=[x], name='sample_cnn')


def res_conv_block(x,filters,strides,cfg,name):
  filters1,filters2,filters3 = filters  # 64,64,256
  
  shortcut = Conv2D(filters3,(1,1),strides=strides,use_bias=True,name=f'{name}_scut_conv',
  kernel_regularizer=l2(cfg.weight_decay), kernel_initializer='glorot_uniform')(x)
  shortcut = BatchNormalization(name=f'{name}_scut_norm')(shortcut)
 
  # block a 64
  x = Conv2D(filters1,(1,1),strides=strides,use_bias=True,name=f'{name}_conva',
  kernel_regularizer=l2(cfg.weight_decay), kernel_initializer='glorot_uniform')(x)
  x = BatchNormalization(name=f'{name}_bna')(x)
  x = Activation('relu',name=f'{name}_relua')(x)
  # block b 64
  x = Conv2D(filters2,(3,3),padding='same',use_bias=True,name=f'{name}_convb',
  kernel_regularizer=l2(cfg.weight_decay), kernel_initializer='glorot_uniform')(x)
  x = BatchNormalization(name=f'{name}_bnb')(x)
  x = Activation('relu',name=f'{name}_relub')(x)
  # block c 256
  x = Conv2D(filters3,(1,1),use_bias=True,name=f'{name}_convc',
  kernel_regularizer=l2(cfg.weight_decay), kernel_initializer='glorot_uniform')(x)
  x = BatchNormalization(name=f'{name}_bnc')(x)

  x = Add(name=f'{name}_scut')([shortcut, x])
  x = Activation('relu', name=f'{name}_relu1')(x)
  return x


def ResCNN(inputshape,cfg):
  """Build a SampleCNN model."""
  # Variable-length input for feature visualization.
  # x_in = Input(shape=(None, 1), name='input')
  x_in = Input(inputshape,name='input')
  # RES block
  num_features = 512
  x = Conv1D(num_features, kernel_size=3, strides=3, padding='same', use_bias=True,
             kernel_regularizer=l2(cfg.weight_decay), kernel_initializer=taejun_uniform(scale=1.), name='conv0')(x_in)
  
  shortcut = BatchNormalization(name='norm0')(x)
  x = Activation('relu', name='relu0')(shortcut)  #(None,512)
  if cfg.amplifying_ratio:
    x = Multiply(name=f're_scale')([x, squeeze_excitation(x, cfg.amplifying_ratio, name='re')])

  x = Add(name=f're_scut')([shortcut, x])
  x = Activation('relu', name=f're_relu')(x)
  x = MaxPool1D(pool_size=3, name=f're_pool')(x)


  x = Lambda(lambda y: K.expand_dims(y,-1),name='expand_dim')(x)
 
  #  ResCNN
  x = Conv2D(3,(3,3),strides=(3,3),padding='same',name='conv1')(x)
  x = BatchNormalization(name='norm1')(x)
  x = Activation('relu', name='relu1')(x)
  # x = MaxPool2D((3,3),strides=(3,3),padding='same',name='pool1')(x)

  x = Conv2D(3,(3,3),strides=(3,3),padding='same',name='conv2')(x)
  x = BatchNormalization(name='norm2')(x)
  x = Activation('relu', name='relu2')(x)

  x = MaxPool2D((3,3),strides=(3,3),padding='same',name='pool2')(x)
  
  # x = res_conv_block(x,[32,32,128],strides=(3,3),cfg=cfg,name='block2')

  x = res_conv_block(x,[64,64,256],strides=(2,2),cfg=cfg,name='block3')
  x = res_conv_block(x,[64,64,256],strides=(2,2),cfg=cfg,name='block4')
  x = res_conv_block(x,[64,64,256],strides=(2,2),cfg=cfg,name='block5')
  # x = res_conv_block(x,[64,64,256],strides=(2,2),cfg=cfg,name='block6')
  
  x = res_conv_block(x,[128,128,512],strides=(2,2),cfg=cfg,name='block6')
  x = res_conv_block(x,[128,128,512],strides=(2,2),cfg=cfg,name='block7')
  x = res_conv_block(x,[128,128,512],strides=(2,2),cfg=cfg,name='block8')
  # x = res_conv_block(x,[128,128,512],strides=(2,2),cfg=cfg,name='block9')


  # 减少维数
  x = Lambda(lambda y: K.mean(y, axis=[1,2]), name='avgpool')(x)

  # The final two FCs.
  x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform', name='final_fc')(x)
  x = BatchNormalization(name='final_norm')(x)
  x = Activation('relu', name='final_relu')(x)
  if cfg.dropout > 0.:
    x = Dropout(cfg.dropout, name='final_drop')(x)
  
  x = Dense(cfg.num_classes, kernel_initializer='glorot_uniform', name='logit')(x)
  x = Activation(cfg.activation, name='pred')(x)

  return Model(inputs=[x_in], outputs=[x], name='ResCNN')


