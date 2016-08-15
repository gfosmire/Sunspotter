import globes as G
import numpy as np
from keras.callbacks import ModelCheckpoint
import json
import h5py
import os
import cPickle
import sys

"""
This is the model builder for the family of number of sunspot pixel prediction models.
It expects to be run from the commandline and to be passed the name of the model to
be run. The model and the data need to be in the folders specified in the globes.py
file of filepaths.
"""

sys.path.append(G.NET)
model_filename = sys.argv[1]

if len(model_filename) > 0:
    mod = __import__(model_filename)

# Parameters
nb_epoch = 25
batch_size = 128

# This actually builds the model
model, model_name = mod.build_model()

# Makes the folder that the model will be saved to
model_dir = G.MOD + model_name + '/'
os.mkdir(model_dir)
temp_path = model_dir + 'temp_model.hdf5'

# Fetch the data
print 'Starting data fetching'
image_lst = os.listdir(G.IMG)
mask_lst = os.listdir(G.MSK)
image_lst.sort()
mask_lst.sort()

# Build the arrays
X = np.zeros((len(image_lst),1,128,128), dtype=np.float32)
for i, image in enumerate(image_lst):
    X[i,0] = np.load(G.IMG+image)/4000.

y = np.zeros(len(mask_lst), dtype=np.float32)
for i, mask in enumerate(mask_lst):
    y[i] = np.load(G.MSK+mask).sum()

print 'Arrays built'

# Shuffle the arrays
indices = np.arange(len(image_lst))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# This updates the temp model if loss improved
checkpointer = ModelCheckpoint(filepath=temp_path, verbose=1, save_best_only=True)

# Train and validate the model
output = model.fit(
            X,
            y,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[checkpointer],
            validation_split=0.1)

# Save the model, history, and parameters
model.save(model_dir + 'final_model.hdf5')

hist = output.history
params = output.params

with open(model_dir + 'history.json', 'wb') as h:
    json.dump(hist, h)
with open(model_dir + 'params.json', 'wb') as p:
    json.dump(params, p)
