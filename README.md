# Sunspotter

##Project Summary
The goal of this project is build and train a convolutional neural net (CNN) to detect sunspots in images of the sun. This was done in two stages: first I made a CNN to make a binary prediction of if an image contained a sunspot. Second I made another CNN that tries to determine where the sunspots are.

##Repo Structure
* model - Contains the stored models as hdf5 files, as well as jsons of parameters and model training history
* src - Contains scripts for model building, training, and validation, as well as data processing
  * pymodels - Contains all of the python model files


##Project Details
This is my Capstone Project for the Galvanize Data Science Immersive program. The neural nets are built using the Keras package with Theano as the backend. More information about Keras can be found at www.keras.io. The data for this project comes from The Precision Solar Photometric Telescope (PSPT), which takes large (2048x2048) images of the sun in 5 different wavelengths several times a day. The data is available at http://lasp.colorado.edu/pspt_access/ in the form of FITS files. In addition to the images of the sun, they also provide mask images of the same size that specify what each pixel represents using a 0-8, with 0 meaning empty space, 2-6 meaning different activity on the sun, and 7 and 8 meaning sunspot. The masks were generated using a semi-empirical model, which can be read about at http://iopscience.iop.org/article/10.1086/307258/pdf. Because not every image had a corresponding mask, I ended up only being able to use one of the wavelength image types, because it always had a mask.

###Acquiring the Data
The LASP site only allows you to download 200MB from them at a time, so I wrote a simple script to do a batch download of all of the image and mask pairs.

###Processing the Data
The images and masks come as FITS files, so I used the Astropy python package to open them and extract the raw numpy arrays. Because the raw arrays are far too large to train a CNN on, I took each 2048x2048 array and tiled it into 256 128x128 arrays. The masks contain more information than I need for this project, so I made my own binary masks with 0 meaning not sunspot and 1 meaning sunspot using where their mask arrays were 7 or 8. Because I only care about tiles that contain the sun, I threw out any tiles that contained only empty space. I then used these arrays to populate the folders that would be used to train the CNNs.

###Building the Models
All of the models that I built are based on Keras' Sequential model. They were built by adding one layer at a time, specifying the kind of layer, and its parameters. Most of the models were primarily built using 2DConvolution, MaxPool2D, and Dense layers. More details can be found by looking at the models in the pymodels folder.

###Training and Validating the Models
Because GPUs are much more efficient at processing image data than CPUs, all of the model training and validating was done on an Amazon Web Services (AWS) GPU instance. 
