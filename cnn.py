
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# refer CNN explainer for the model
# Initialising the CNN
classifier = Sequential()
# seqeuntial creates a pooling eg. Conv-ReLu-MaxPooling repeat

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# 32 is a kernel/filter i.e extract 32 features can be changed is subsequent layers
#(3,3) = kernel sixe(3*3) 
#input_shape = Dimensions of pictures should be constant 3  means has 3 color channel( red, blue green)
#Should be mentioned only once
#activation - experiement with different functions

# apply 64 filter
classifier,add(Conv2D(64,(3,3), activation  = ' relu))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer

classifier.add(Conv2D(20,(5,5), activation = 'relu')
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection - Feed Forward
#Dense means hidden layer, unit = no of neurons, repeat the code  to increase the hidden layer, Below has 2 hidden layers. Takes on 128 data works like a step size
               
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# rescaling the images

train_datagen = ImageDataGenerator(rescale = 1./255,# makes black and white images
                                   shear_range = 0.2,
                                   zoom_range = 0.2, # aspect ratio
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

#Create train and vadidation test data               
training_set = train_datagen.flow_from_directory('/Users/sudhanshukumar/Downloads/PetImages-20190330T113641Z-001/PetImages',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/Users/sudhanshukumar/Downloads/PetImages-20190330T113641Z-001/PetImages',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,# no of times the image is sent
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 2000)# after 128*2000 validate the image

classifier.save("model.h5")# generates the file and save it to deploy it
print("Saved model to disk")

# Part 3 - Making new predictions




import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/sudhanshukumar/Downloads/cat.11.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)