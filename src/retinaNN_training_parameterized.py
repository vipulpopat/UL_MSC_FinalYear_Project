###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
#import ConfigParser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training


def get_encoder_part(inputs, numberOfFilters, addPoolingLayer, pooling_type, kernel):
    print(f'get_encoder_part() {inputs} {numberOfFilters} {addPoolingLayer} {kernel} {pooling_type}')
    conv1 = Conv2D(numberOfFilters, kernel, activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(numberOfFilters, kernel, activation='relu', padding='same',data_format='channels_first')(conv1)
    if (addPoolingLayer):
        if pooling_type == 1:
            pool1 = MaxPooling2D((2, 2))(conv1)
        else:
            pool1 = AveragePooling2D((2, 2))(conv1)
        return pool1, conv1
    else:
        return conv1, conv1

def get_decoder_part(inputs, numberOfFilters, encoder, kernel):
    print(f'get_decoder_part() {inputs} {numberOfFilters} {encoder} {kernel}')
    up = UpSampling2D(size=(2, 2))(inputs)
    up = concatenate([encoder, up],axis=1)
    conv = Conv2D(numberOfFilters, kernel, activation='relu', padding='same',data_format='channels_first')(up)
    conv = Dropout(0.2)(encoder)
    conv = Conv2D(numberOfFilters, kernel, activation='relu', padding='same',data_format='channels_first')(conv)
    return conv

#Define the neural network
def get_unet(n_ch,patch_height,patch_width, network_depth, number_filters, pooling_types, kernels, model_optimizer):
    print(f'get_unet() {n_ch} {patch_height} {patch_width} {network_depth} {number_filters} {kernels} {pooling_types} {model_optimizer}')
    inputs = Input(shape=(n_ch,patch_height,patch_width))

    if network_depth != len(number_filters):
        raise ValueError('network_depth and number_filters count should be the same')

    if (model_optimizer == 1):
        optimizer_string = 'sgd'
    elif (model_optimizer == 2):
        optimizer_string = 'adam'
    elif (model_optimizer == 3):
        optimizer_string = 'adamax'
    else:
        optimizer_string = 'nadam'

    encoders = []
    network = inputs

    for d in range(1, network_depth+1):
        kernel_tuple = get_kernel_tuple(int(kernels[d-1]))
        pooling_type = pooling_types[d-1]
        network, encoder = get_encoder_part(network, int(number_filters[d-1]), d != network_depth, pooling_type, kernel_tuple)
        encoders.append(encoder)

    for d in range(network_depth-1, 0, -1):
        kernel_tuple = get_kernel_tuple(int(kernels[d-1]))
        network = get_decoder_part(network, int(number_filters[d-1]), encoders[d-1], kernel_tuple)

    network = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(network)
    network = core.Reshape((2,patch_height*patch_width))(network)
    network = core.Permute((2,1))(network)
    ############
    network = core.Activation('softmax')(network)

    model = Model(inputs=inputs, outputs=network)        

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=optimizer_string, loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def get_kernel_tuple(kernel_type):
    print(f'get_kernel_tuple({kernel_type})')
        # get the kernel tuple
    kernel_tuple = (3,3)

    if (kernel_type == 1):
        kernel_tuple = (3,3)
    elif (kernel_type == 2):
        kernel_tuple = (5,5)
    elif (kernel_type == 3):
        kernel_tuple = (7,7)
    else:
        kernel_tuple = (9,9)

    return kernel_tuple

#========= Load settings from Config file
config = configparser.RawConfigParser()
#config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
network_depth = int(config.get('training settings','network_depth'))
print(f'network_depth={network_depth}')
number_filters = config.get('training settings','number_filters').split(',')
print(f'number_filters={number_filters}')
pooling_types = config.get('training settings','pooling_types').split(',')
print(f'pooling_types={pooling_types}')
kernels = config.get('training settings','kernels').split(',')
print(f'kernels={kernels}')
optimizer = int(config.get('training settings','optimizer'))
print(f'optimizer={optimizer}')

model = get_unet(n_ch, patch_height, patch_width, network_depth, number_filters, pooling_types, kernels, optimizer)  #the U-net model

print( "Check: final output of the network:")
print( f'Model output shape={model.output_shape}')
print('Model Summary', model.summary())
print(f'Model train imgs={patches_imgs_train.shape}, mask={patches_masks_train.shape}')

plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_acc', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
