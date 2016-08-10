import globes as G
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import json
import h5py
import os
import cPickle
import sys

sys.path.append(G.NET)
model_filename = sys.argv[1]

if len(model_filename) > 0:
    mod = __import__(model_filename)

# parameters
nb_epoch = 25
batch_size = 128

model, model_name = mod.build_model()

# saves the output in the model_dir
model_dir = G.MOD + model_name + '/'
os.mkdir(model_dir)
temp_path = model_dir + 'temp_model.hdf5'

#update the model if stuff gets better
checkpointer = ModelCheckpoint(filepath=temp_path, verbose=1, save_best_only=True)

# this actually fits the model
output = model.fit(
            X,
            y,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[checkpointer],
            validation_split=0.1,
            )

model.save(model_dir + 'final_model.hdf5')

hist = output.history
params = output.params

with open(model_dir + 'history.json', 'wb') as h:
    json.dump(hist, h)
with open(model_dir + 'params.json', 'wb') as p:
    json.dump(params, p)
