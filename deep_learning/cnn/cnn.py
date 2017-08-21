# Convolutional Neural Network (CNN) in Python

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 

# Part 1 - Building the CNN
classifier = Sequential()

# Step 1 - Convolution
# nb_filters = # of feature maps(commonly 32), # rows,columns in feature detector table
# input_shape = images don't have same format so they should be converted to a fixed size

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Max pooling
# size of pool = 2x2 dimensions (reduce size of feature maps)
classifier.add(MaxPooling2D(pool_size = (2,2)))

# add a 2nd Convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Full Connection of the network
# add hidden layer
# units = # nodes in hidden layer (choose from experimentation)
classifier.add(Dense(activation="relu", units=128))
# add output layer
# sigmoid AF for binary outcome
classifier.add(Dense(activation="sigmoid", units=1))

# Compiling the CNN
# optimizer - stochastic gradient descent (adam)
# loss function - binary cross_entropy (logarithmic loss)
# if > 2 outcomes - use categorical_crossentropy
classifier.compile(optimizer = 'adam' , loss = 
                   'binary_crossentropy', metrics = ['accuracy'] )

# Fitting CNN and testing 
from keras.preprocessing.image import ImageDataGenerator

# image augmentation - generate new images from dataset
# with random transformations to reduce overfitting 
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True)

# preprocess images of testing set
test_datagen = ImageDataGenerator(rescale = 1./255)

# apply image augmentation and resize images;
# create batches of 32 images
# increase size of images can increase accuracy
train_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Fit CNN to Training set and test performance on Testing set
# steps per epoch = # images in training set
# validation set = # images in testing set
#from PIL import Image
classifier.fit_generator(train_set, steps_per_epoch = 8000,
                        epochs = 10,
                        validation_data = test_set,
                        validation_steps = 2000)
'''
classifier.fit_generator(train_set, samples_per_epoch = 8000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 2000)
'''
