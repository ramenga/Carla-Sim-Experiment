#train a model from data collected
import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16,ResNet50,MobileNetV2

from model_dpv3 import Deeplabv3


from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#helper class to define input shape and generate training images given image paths & steering angles
from utils2_resnetDeeplab import INPUT_SHAPE, batch_generator
#for command line arguments
import argparse
#for reading files
import os

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['center', 'left', 'right']].values
    #X = data_df[['center']].values
    #and our steering commands as our output data
    y = data_df['steering'].values
    print("pandas",y.shape)
    print(type(y))

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    #X_train,etc. is just a list of image file paths, not real data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    print("SHAPES")
    print(X_train)
    print(type(X_train))
    print(y_train.shape)
    print(y_train)

    return X_train, X_valid, y_train, y_valid

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def build_Resnet50_DeepLab(args):

    rs50 = ResNet50(include_top = False, pooling = 'avg', weights='imagenet',
        input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
    for layer in rs50.layers:
        print(layer, layer.trainable)

    for layer in rs50.layers[:]:
        layer.trainable = False

    #mbv2 = MobileNetV2(include_top=False)
    '''
    dlv3 =  Deeplabv3(input_shape=(512,512,3),backbone="mobilenetv2", classes=1) 
    for layer in dlv3.layers:
        print(layer, layer.trainable)
    '''

    model = Sequential()
    model.add(rs50)
    model.add(Flatten())
    model.add(Dense(1024,activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='linear'))

    return model



def build_vggcustom_model(args):
    # Initialize the VGG model without the top layers 
    vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
    # Freeze all the layers

    # Freeze all layers of the VGG16
    for layer in vgg_conv.layers[:]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    #Create a new model that starts with the vgg16
    model = Sequential()
    model.add(vgg_conv)
    model.add(Flatten())
    model.add(Dense(1024,activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='linear'))


    return model


def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = keras.Sequential()
    print("ips",INPUT_SHAPE)
    #model.add(Flatten(input_shape=INPUT_SHAPE))
    #model.add(layers.Dense(100, activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=(2,2),activation='elu',padding='same',input_shape=INPUT_SHAPE))
    #model.add(layers.Lambda(lambda x: x/255.0, input_shape=INPUT_SHAPE))
    model.add(layers.Conv2D(32, (5, 5), activation='elu', strides=(2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='elu', strides=(2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='elu', strides=(2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='elu'))
    model.add(layers.Conv2D(16, (3, 3), activation='elu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(300, activation='elu'))
    model.add(layers.Dense(100, activation='elu'))
    model.add(layers.Dense(70, activation='elu'))
    model.add(layers.Dense(1,activation = 'relu'))
    model.summary()

    return model

def build_model_1(args):
    model = keras.Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(layers.Dense(300, activation='elu'))
    model.add(layers.Dense(100, activation='elu'))
    model.add(layers.Dense(70, activation='elu'))
    model.add(layers.Dense(1,activation = 'relu'))
    model.summary()

    return model

def build_model_2(args):
    model = Sequential()

    model.add(Lambda(lambda x: x /127.5 - 1., input_shape=INPUT_SHAPE))
    model.add(Conv2D(32, (3,3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(256,activation='elu'))
    model.add(Dense(64,activation='elu'))
    model.add(Dense(1,activation='linear'))
    #model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer='Adadelta',metrics=['accuracy'])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    #model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    '''
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, False),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_queue_size=2,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        callbacks=[checkpoint],
                        verbose=1)
	
	'''
	#model.fit(X_train,y_train,validation_split=0.1,batch_size=10,epochs=40,shuffle=True,verbose=1
    #model.fit(X_train,y_train,batch_size=2,epochs=2,verbose=1,steps_per_epoch=5000)
    model.fit(batch_generator(args.data_dir, X_train, y_train, args.batch_size, False),batch_size=32,epochs=5,shuffle=True,
    	verbose=1,steps_per_epoch=7000)
    
    return model

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.1)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=1)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=200)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=10)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=0.001)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)

    #build model
    #model = build_model_2(args)


    
    if os.path.isfile('resDeep_autopilot.h5') is True:
    	model = load_model('resDeep_autopilot.h5')
    else:
        print("MODEL Not Found")
        model = build_Resnet50_DeepLab(args)

    # Show model
    model.summary()

    
    #train model on data, it saves as model.h5 
    model = train_model(model, args, *data)

    model.save('resDeep_autopilot.h5')

if __name__ == '__main__':
    main()

