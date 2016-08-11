import globes as G
import numpy as np
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

# fetch the data
print 'Starting data fetching'
image_lst = os.listdir(G.IMG)
mask_lst = os.listdir(G.MSK)
image_lst.sort()
mask_lst.sort()

# build the arrays
print 'Building arrays'
X = np.zeros((len(image_lst),1,128,128), dtype=np.float32)
for i, image in enumerate(image_lst):
    X[i,0] = np.load(G.IMG+image)/4000.

y = np.zeros((len(mask_lst),1,128,128), dtype=np.float32)
for i, mask in enumerate(mask_lst):
    y[i] = np.load(G.MSK+mask)

print 'Arrays built'

# shuffle the arrays
indices = np.arange(len(image_lst))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# update the model if stuff gets better
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
