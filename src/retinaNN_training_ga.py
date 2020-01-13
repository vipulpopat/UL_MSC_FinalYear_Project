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


def get_encoder_part(inputs, numberOfFilters, addPoolingLayer, pooling_type):
    print('get_encoder_part()' + str(inputs) + str(numberOfFilters) + str(addPoolingLayer))
    conv1 = Conv2D(numberOfFilters, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(numberOfFilters, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    if (addPoolingLayer):
        if pooling_type == 1:
            pool1 = MaxPooling2D((2, 2))(conv1)
        else:
            pool1 = AveragePooling2D((2, 2))(conv1)
        return pool1, conv1
    else:
        return conv1, conv1

def get_decoder_part(inputs, numberOfFilters, encoder):
    print('get_decoder_part()' + str(inputs) + str(numberOfFilters) + str(encoder))
    up = UpSampling2D(size=(2, 2))(inputs)
    up = concatenate([encoder, up],axis=1)
    conv = Conv2D(numberOfFilters, (3, 3), activation='relu', padding='same',data_format='channels_first')(up)
    conv = Dropout(0.2)(encoder)
    conv = Conv2D(numberOfFilters, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv)
    return conv

#Define the neural network
def get_unet(n_ch,patch_height,patch_width, network_depth, number_filters, pooling_type):
    print('get_unet() '+str(n_ch) + str(patch_height) + str(patch_width) + str(network_depth) + str(number_filters))
    inputs = Input(shape=(n_ch,patch_height,patch_width))

    if network_depth != len(number_filters):
        raise ValueError('network_depth and number_filters count should be the same')

    encoders = []
    network = inputs

    for d in range(1, network_depth+1):
        network, encoder = get_encoder_part(network, int(number_filters[d-1]), d != network_depth, pooling_type)
        encoders.append(encoder)

    for d in range(network_depth-1, 0, -1):
        network = get_decoder_part(network, int(number_filters[d-1]), encoders[d-1])

    network = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(network)
    network = core.Reshape((2,patch_height*patch_width))(network)
    network = core.Permute((2,1))(network)
    ############
    network = core.Activation('softmax')(network)

    model = Model(inputs=inputs, outputs=network)        

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

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

n=2
def get_unet_params(binary_list):
    d, f, p = 1, [], 1
    final = [binary_list[i * n:(i + 1) * n] for i in range((len(binary_list) + n - 1) // n )]
    if (compare(final[0],[0,0])):
        d=1
    elif (compare(final[0],[0,1])):
        d=2
    elif (compare(final[0],[1,0])):
        d=3
    elif (compare(final[0],[1,1])):
        d=4
        
    for i in range(1,d+1):
        f.append(get_filter_count(final[i]))
        
    if sum(final[5]) == 0:
        p = 1
    else:
        p = 2
        
    return d,f,p
    
def get_filter_count(cb):
    if (compare(cb,[0,0])):
        return 16
    elif (compare(cb,[0,1])):
        return 32
    elif (compare(cb,[1,0])):
        return 64
    elif (compare(cb,[1,1])):
        return 256
    
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

NETWORK_DEPTH=4+1
INDIVIDUAL_SIZE = NETWORK_DEPTH*2 + 1

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=INDIVIDUAL_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def eval_model_loss_function(individual):
    print(f'GA------------Individual = {individual}, parameters={get_unet_params(individual)}')
    d,f,p = get_unet_params(individual)
    model1 = get_unet(n_ch, patch_height, patch_width, d, f, p)  #the U-net model
    print('GA Model output shape = ',model1.output_shape)
    print('GA Model Summary', model1.summary())
    print(f'GA Model train imgs={patches_imgs_train.shape}, mask={patches_masks_train.shape}')
    history = model1.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1)
    fitness = min(history.history['val_loss'])
    print(f'GA------------Depth={d}, Filters = {f}, pooling_type={p}, fitness={fitness}')
    return fitness,

toolbox.register("evaluate", eval_model_loss_function)

pop = toolbox.population(n=N_individuals)

CXPB, MUTPB, NGEN = 0.5, 0.2, N_generations
print( "GA------------Starting the Evolution Algorithm...")

for g in range(NGEN):
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

    print(f'GA------------\tEvaluated {len(pop)} individuals')

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print(f"GA------------\tMin {min(fits)}")
    print(f"GA------------\tMax {max(fits)}")
    print(f"GA------------\tAvg {mean}")
    print(f"GA------------\tStd {std}")
        
top = tools.selBest(pop, k=1)

print(f'GA------------U net configuration = {get_unet_params(top[0])}')
best_d, best_f, best_p = get_unet_params(top[0])

#=============================================================


model = get_unet(n_ch, patch_height, patch_width, best_d, best_f, best_p)  #the U-net model


print( "Check: final output of the network:")
print( model.output_shape)
plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased


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
