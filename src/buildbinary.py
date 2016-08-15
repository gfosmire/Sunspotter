import globes as G
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import json
import h5py
import os
import cPickle
import sys

"""
This is the model builder for the family of binary classification models. It expects
to be run from the commandline and to be passed the name of the model to be run.
The model and the data need to be in the folders specified in the globes.py file
of filepaths.
"""

sys.path.append(G.NET)
model_filename = sys.argv[1]

if len(model_filename) > 0:
    mod = __import__(model_filename)

# Parameters
nb_epoch = 25
batch_size = 128
img_height, img_width = 128, 128

# This is the configuration that will be used for training and validation
train_datagen = ImageDataGenerator(
        rescale = 1./4000,
        fill_mode = 'constant')

val_datagen = ImageDataGenerator(
        rescale = 1./4000,
        fill_mode = 'constant')

train_generator = train_datagen.flow_from_directory(
        G.BTRN,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        color_mode = 'grayscale')

val_generator = val_datagen.flow_from_directory(
        G.BVAL,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        color_mode = 'grayscale')

# This actually builds the model
model, model_name = mod.build_model()

# Makes the folder that the model will be saved to
model_dir = G.MOD + model_name + '/'
os.mkdir(model_dir)
temp_path = model_dir + 'temp_model.hdf5'

# This updates the temp model if loss improved
checkpointer = ModelCheckpoint(filepath=temp_path, verbose=1, save_best_only=True)

# Train and validate the model
output = model.fit_generator(
            train_generator,
        	samples_per_epoch=train_generator.N,
        	nb_epoch=nb_epoch,
        	validation_data=val_generator,
        	nb_val_samples=val_generator.N,
            callbacks=[checkpointer])

# Save the model, history, and parameters
model.save(model_dir + 'final_model.hdf5')

hist = output.history
params = output.params

with open(model_dir + 'history.json', 'wb') as h:
    json.dump(hist, h)
with open(model_dir + 'params.json', 'wb') as p:
    json.dump(params, p)
