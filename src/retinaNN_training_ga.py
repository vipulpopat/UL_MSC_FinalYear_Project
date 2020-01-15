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
import random
import collections

from deap import base, creator, tools, algorithms

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, AveragePooling2D
from keras import optimizers
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
N_generations = int(config.get('ga settings', 'generations'))
N_individuals = int(config.get('ga settings', 'individuals'))
CXPB = float(config.get('ga settings', 'cxpb'))
MUTPB = float(config.get('ga settings', 'mutpb'))


#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)

patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

#====================the GA Code is being kept in here========================
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def get_unet_params(binary_list):
    d, f, k, o, p = 1, [], [], 1, []
    
    depth_bin = binary_list[0:2]
    if (compare(depth_bin,[0,0])):
        d=1
    elif (compare(depth_bin,[0,1])):
        d=2
    elif (compare(depth_bin,[1,0])):
        d=3
    elif (compare(depth_bin,[1,1])):
        d=4
        
    filter_binary = binary_list[2:14]        
    k_b = binary_list[14:22]
    o_b = binary_list[22:24]
    p_b = binary_list[24:28]

    for i in range(0,d):
        f.append(get_filter_count(filter_binary[i*3:(i*3+3)]))
        k.append(get_bin_count(k_b[i*2:(i*2+2)]))
        p.append(get_bin_count(p_b[i:(i+1)]))
        
    o = get_bin_count(o_b)
    
    return d,f,k,o,p
    
def get_filter_count(cb):
    if (compare(cb,[0,0,0])):
        return 8
    elif (compare(cb,[0,0,1])):
        return 16
    elif (compare(cb,[0,1,0])):
        return 32
    elif (compare(cb,[0,1,1])):
        return 64
    elif (compare(cb,[1,0,0])):
        return 96
    elif (compare(cb,[1,0,1])):
        return 128
    elif (compare(cb,[1,1,0])):
        return 192
    elif (compare(cb,[1,1,1])):
        return 256  
    
def get_bin_count(final):
    if (compare(final,[0,0]) or (sum(final) == 0)):
        return 1
    elif (compare(final,[0,1]) or (sum(final) == 1)):
        return 2
    elif (compare(final,[1,0])):
        return 3
    else:
        return 4
        
creator.create('FitnessMin', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

INDIVIDUAL_SIZE = 2 + (4*3) + (4*2) + 2 + (4)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=INDIVIDUAL_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def eval_model_loss_function(individual):
    print(f'GA------------Individual = {individual}, parameters={get_unet_params(individual)}')
    d,f,k,o,p = get_unet_params(individual)
    model1 = get_unet(n_ch, patch_height, patch_width, d, f, p, k, o)  #the U-net model
    print('GA------------Model Summary', model1.summary())
    history = model1.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1)
    fitness = max(history.history['val_acc'])
    fitness_vector=history.history['val_acc']
    print(f'GA------------fitness vector={fitness_vector}')
    print(f'GA------------Depth={d}, Filters = {f}, pooling_type={p}, kernels={k}, optimizer={o}, fitness={fitness}')
    return fitness,

toolbox.register("evaluate", eval_model_loss_function)

pop = toolbox.population(n=N_individuals)

print( "GA------------Starting the Evolution Algorithm...")

for g in range(N_generations):
    print(f"GA-------------- Generation {g} --")

    # Select the next genereation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print(f'GA------------Generation {g}---\tEvaluated {len(pop)} individuals')

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    top = tools.selBest(pop, k=1)
    print(f"GA------------Generation {g}\tMin {min(fits)}")
    print(f"GA------------Generation {g}\tMax {max(fits)}")
    print(f"GA------------Generation {g}\tAvg {mean}")
    print(f"GA------------Generation {g}\tStd {std}")
    print(f"GA------------Generation {g}\tBest UNet Configuration {get_unet_params(top[0])}")
    print(f"GA------------End of Generation {g}")
        
top = tools.selBest(pop, k=1)

print(f'GA------------Across Generations, U net configuration = {get_unet_params(top[0])}')
best_d, best_f, best_p = get_unet_params(top[0])

#=============================================================


model = get_unet(n_ch, patch_height, patch_width, best_d, best_f, best_p)  #the U-net model


print( "Check: final output of the network:")
print( model.output_shape)
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

model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


















#
