import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop

def pick_SOAT(data, leakage_model, length, metric, loss, model='MLP', model_size=64):
    if model == 'CNN':
        if data=='ASCAD' and leakage_model=='HW':
            batch_size = 50
            epoch = 50
            return ascad_f_hw_cnn_rs(length, metric, loss), batch_size, epoch
        elif data=='ASCAD' and leakage_model=='ID':
            batch_size = 50
            epoch = 50
            return ascad_f_id_cnn(length, metric, loss), batch_size, epoch
        elif data=='ASCAD_rand' and leakage_model=='HW':
            batch_size = 128
            epoch = 50
            return ascad_r_hw_cnn_rs(length, metric, loss), batch_size, epoch
        elif data=='ASCAD_rand' and leakage_model=='ID':
            batch_size = 128
            epoch = 50
            return ascad_r_id_cnn(length, metric, loss), batch_size, epoch
        elif data=='CHES_CTF' and leakage_model=='HW':
            batch_size = 128
            epoch = 50
            return ches_ctf_hw_cnn(length, metric, loss), batch_size, epoch           
    elif model == 'MLP':
        if data=='ASCAD' and leakage_model=='HW':
            batch_size = 32
            epoch = 10
            return ascad_f_hw_mlp(length, metric, loss), batch_size, epoch
        elif data=='ASCAD' and leakage_model=='ID':
            batch_size = 32
            epoch = 10
            return ascad_f_id_mlp(length, metric, loss), batch_size, epoch
        elif data=='ASCAD_rand' and leakage_model=='HW':
            batch_size = 32
            epoch = 10
            return ascad_r_hw_mlp(length, metric, loss), batch_size, epoch
        elif data=='ASCAD_rand' and leakage_model=='ID':
            batch_size = 32
            epoch = 10
            return ascad_r_id_mlp(length, metric, loss), batch_size, epoch
        elif data=='CHES_CTF' and leakage_model=='HW':
            batch_size = 32
            epoch = 10
            return ches_ctf_hw_mlp(length, metric, loss), batch_size, epoch
    elif model == 'model_size':
        batch_size = 200
        epoch = 75
        if leakage_model == 'HW':
            return cnn_best(length, metric, loss, classes=9, unit=model_size), batch_size, epoch 
        else:
            return cnn_best(length, metric, loss, classes=256, unit=model_size), batch_size, epoch 
              

################ SOAT MODELS#####################
# epoch 50
def ascad_f_hw_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(16, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_f_hw_cnn_rs(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_f_hw_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(496, activation='relu')(img_input)
    x = Dense(496, activation='relu')(x)
    x = Dense(136, activation='relu')(x)
    x = Dense(288, activation='relu')(x)
    x = Dense(552, activation='relu')(x)
    x = Dense(408, activation='relu')(x)
    x = Dense(232, activation='relu')(x)
    x = Dense(856, activation='relu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ascad_f_hw_mlp_1(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(1024, activation='relu')(img_input)
    x = Dense(1024, activation='relu')(x)
    x = Dense(760, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(704, activation='relu')(x)
    x = Dense(1016, activation='relu')(x)
    x = Dense(560, activation='relu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=1e-5)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ascad_f_id_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_f_id_cnn_rs(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 75, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Conv1D(2, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4)(x)
    x = Conv1D(8, 2, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(2, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_f_id_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(160, activation='relu')(img_input)
    x = Dense(160, activation='relu')(x)
    x = Dense(624, activation='relu')(x)
    x = Dense(776, activation='relu')(x)
    x = Dense(328, activation='relu')(x)
    x = Dense(968, activation='relu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0001)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ascad_f_id_mlp_1(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(480, activation='elu')(img_input)
    x = Dense(480, activation='elu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=1e-5)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

# epoch:50/batch_size:400
def ascad_r_hw_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(8, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_r_hw_cnn_rs(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 50, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(25, strides=25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_r_hw_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(200, activation='elu')(img_input)
    x = Dense(200, activation='elu')(x)
    x = Dense(304, activation='elu')(x)
    x = Dense(832, activation='elu')(x)
    x = Dense(176, activation='elu')(x)
    x = Dense(872, activation='elu')(x)
    x = Dense(608, activation='elu')(x)
    x = Dense(512, activation='elu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ascad_r_hw_mlp_1(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(448, activation='elu')(img_input)
    x = Dense(448, activation='elu')(x)
    x = Dense(512, activation='elu')(x)
    x = Dense(168, activation='elu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

# epoch:50/batch_size:400
def ascad_r_id_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(75, strides=75)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(2, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_r_id_cnn_rs(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(100, strides=75)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(30, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(2, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    # optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer='adam', metrics=metric)
    model.summary()
    return model

def ascad_r_id_mlp(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(256, activation='elu')(img_input)
    x = Dense(256, activation='elu')(x)
    x = Dense(296, activation='elu')(x)
    x = Dense(840, activation='elu')(x)
    x = Dense(280, activation='elu')(x)
    x = Dense(568, activation='elu')(x)
    x = Dense(672, activation='elu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ascad_r_id_mlp_1(length, metric, loss):
    img_input = Input(shape=(length,))
    x = Dense(664, activation='elu')(img_input)
    x = Dense(664, activation='elu')(x)
    x = Dense(624, activation='elu')(x)
    x = Dense(816, activation='elu')(x)
    x = Dense(624, activation='elu')(x)
    x = Dense(256, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ches_ctf_hw_cnn_1(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(4, 100, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(4, strides=4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ches_ctf_hw_cnn(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Conv1D(2, 2, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(7, strides=7)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = Adam(lr=5e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ches_ctf_hw_mlp_1(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Flatten(name='flatten')(img_input)
    x = Dense(696, activation='elu')(x)
    x = Dense(696, activation='elu')(x)
    x = Dense(168, activation='elu')(x)
    x = Dense(184, activation='elu')(x)
    x = Dense(848, activation='elu')(x)
    x = Dense(568, activation='elu')(x)
    x = Dense(328, activation='elu')(x)
    x = Dense(584, activation='elu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def ches_ctf_hw_mlp(length, metric, loss):
    img_input = Input(shape=(length, 1))
    x = Flatten(name='flatten')(img_input)
    x = Dense(192, activation='elu')(x)
    x = Dense(192, activation='elu')(x)
    x = Dense(616, activation='elu')(x)
    x = Dense(248, activation='elu')(x)
    x = Dense(440, activation='elu')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=0.001)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

def cnn_best(length, metric, loss, classes=9, unit=64):
    # From VGG16 design
    input_shape = (length, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(unit*1, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(unit*2, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(unit*4, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(unit*8, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(unit*8, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(unit*64, activation='relu', name='fc1')(x)
    x = Dense(unit*64, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    model.summary()
    return model

# def mlp_best(length, metric, loss, lr=0.00001, node=200, layer_nb=6, initializer='glorot_uniform'):
#     model = Sequential()
#     model.add(Dense(node, input_dim=length, activation='relu', kernel_initializer=initializer))

#     for i in range(layer_nb - 2):
#         model.add(Dense(node, activation='relu', kernel_initializer=initializer))

#     model.add(Dense(classes, activation='softmax'))
#     optimizer = RMSprop(lr=lr)
#     model.compile(loss=loss, optimizer=optimizer, metrics=metric)
#     model.summary()
#     return model