from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam

def build_model():
    model_name = 'basic3'

    model = Sequential()
    model.add(Convolution2D(64, 5, 5, input_shape=(1,128,128), init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, init='he_normal'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy']
                 )

    return model, model_name
