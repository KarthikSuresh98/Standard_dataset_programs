from keras.datasets import cifar10
from keras.layers import Conv2D , MaxPooling2D , Dropout , Flatten , Dense , Input
from keras.models import Sequential
from keras import utils as np_utils
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


(X_train , y_train) , (X_test , y_test) = cifar10.load_data()
X_train = X_train/255.0
X_test = X_test/255.0
y_train = np.ravel((y_train))
y_test = np.ravel((y_test))
y_train_m = np_utils.to_categorical(y_train , num_classes  = 10)

def model(X_train , y_train , X_test):
    model = Sequential()
    model.add(Conv2D(300 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (32,32,3)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(300 , (2,2) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(300 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(300 , (2,2) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(300 , activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100 , activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10 , activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    model.fit(X_train , y_train , batch_size = 128 , epochs = 1)
    y_pred = model.predict(X_test)
    return y_pred

predictions = model(X_train , y_train_m , X_test)
y_res = predictions.argmax(axis = -1)
print(accuracy_score(y_res , y_test))
