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
y_test = np_utils.to_categorical(np.reshape(y_test , (y_test.shape[0] , 1)) , num_classes  = 10) only for function : model
X_test = np.asarray(df)
X_test = np.reshape(X_test , (X_test.shape[0] , 28 , 28 , 1))


def model(X_t , y_t , X_te , y_te):
    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same' , input_shape = (28,28,1)))
    model.add(Conv2D(32 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
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
    score = model.evaluate(X_te , y_te , batch_size = 64)
    print(score)

# model(X_train , y_train , X_test , y_test) model achieves training set accuracy of 96.07% and test set accuracy of 92.94%
