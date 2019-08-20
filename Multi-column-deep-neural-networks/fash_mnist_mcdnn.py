import pandas as pd
import numpy as np
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten , Dropout , Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

loc = r'/home/karthik/Downloads/datasets/fashion_mnist/train.csv'
df = pd.read_csv(loc)
y_train = np.asarray(df['label'])
del df['label']
y_train = np_utils.to_categorical(np.reshape(y_train , (y_train.shape[0] , 1)) , num_classes  = 10)
X_train = np.asarray(df)
X_train = np.reshape(X_train , (X_train.shape[0] , 28 , 28 , 1))
loc2 = r'/home//karthik/Downloads/datasets/fashion_mnist/test.csv'
df = pd.read_csv(loc2)
y_test = np.asarray(df['label'])
del df['label']
X_test = np.asarray(df)
X_test = np.reshape(X_test , (X_test.shape[0] , 28 , 28 , 1))

def model1(X_t , y_t , X_te):
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (28,28,1)))
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    model.fit(X_t , y_t , batch_size = 256 , epochs = 15)
    y_pred = model.predict(X_te)
    return y_pred

def model2(X_t , y_t , X_te):
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (28,28,1)))
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    datagen = ImageDataGenerator(rotation_range = 5)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train , y_train , batch_size = 256) , epochs = 15)
    y_pred = model.predict(X_te)
    return y_pred

def model3(X_t , y_t , X_te):
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (28,28,1)))
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    datagen = ImageDataGenerator(zca_whitening = 5)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train , y_train , batch_size = 256) , epochs = 15)
    y_pred = model.predict(X_te)
    return y_pred

def model4(X_t , y_t , X_te):
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (28,28,1)))
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    datagen = ImageDataGenerator(width_shift_range = 0.1 , height_shift_range = 0.1)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train , y_train , batch_size = 256) , epochs = 15)
    y_pred = model.predict(X_te)
    return y_pred

def model5(X_t , y_t , X_te):
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (28,28,1)))
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    datagen = ImageDataGenerator(horizontal_flip = True)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train , y_train , batch_size = 256) , epochs = 15)
    y_pred = model.predict(X_te)
    return y_pred

y_res1 = model1(X_train , y_train , X_test) + model1(X_train , y_train , X_test) + model1(X_train , y_train , X_test) + model1(X_train , y_train , X_test)
y_res2 = model2(X_train , y_train , X_test)
y_res3 = model3(X_train , y_train , X_test)
y_res4 = model4(X_train , y_train , X_test)
y_res5 = model5(X_train , y_train , X_test)
y_res = (y_res1 + y_res2 + y_res3 + y_res4 + y_res5)/8
Y_res = y_res.argmax(axis = -1)
print(accuracy_score(Y_res , y_test))

#mcdnn implementation achieves training set accuracy of 94-98% and test set accuracy of 94.38%. This is clearly greater than that achieved by single neural network model.
