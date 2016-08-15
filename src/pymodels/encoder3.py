from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam

def build_model():
    model_name = 'encoder3'

    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(1,128,128), init='he_normal', border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(1, 1, 1, init='he_normal'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy']
                 )

    return model, model_name
