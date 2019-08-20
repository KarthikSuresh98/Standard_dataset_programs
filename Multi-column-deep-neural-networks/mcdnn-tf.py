# Yet to be completed


import tensorflow as tf
from keras.datasets import cifar10
from keras import utils as np_utils


(X_train , y_train) , (X_test , y_test) = cifar10.load_data()
X_train = X_train/255.0
X_test = X_test/255.0
y_train = np_utils.to_categorical(y_train , num_classes = 10)
y_test = np_utils.to_categorical(y_test , num_classes = 10)
batch_size  = 128

x = tf.placeholder("float" , [None,32,32,3])
y = tf.placeholder("float" , [None,10])


weights = {"wc1" : tf.get_variable('W1' , shape = (3,3,3,300) , initializer = tf.contrib.layers.xavier_initializer()) ,
               "wc2" : tf.get_variable('W2' , shape = (2,2,300,300) , initializer = tf.contrib.layers.xavier_initializer()) ,
               "wc3" : tf.get_variable('W3' , shape = (3,3,300,300) , initializer = tf.contrib.layers.xavier_initializer()) ,
               "wc4" : tf.get_variable('W4' , shape = (2,2,300,300) , initializer = tf.contrib.layers.xavier_initializer()) ,
               "wd1" : tf.get_variable('W5' , shape = (1200,300) , initializer = tf.contrib.layers.xavier_initializer()) ,
               "wd2" : tf.get_variable('W6' , shape = (300,10) , initializer = tf.contrib.layers.xavier_initializer()) }
biases = {"bc1" : tf.get_variable('B1' , shape = (300) , initializer = tf.contrib.layers.xavier_initializer()),
            "bc2" : tf.get_variable('B2' , shape = (300) , initializer = tf.contrib.layers.xavier_initializer()),
            "bc3" : tf.get_variable('B3' , shape = (300) , initializer = tf.contrib.layers.xavier_initializer()),
            "bc4" : tf.get_variable('B4' , shape = (300) , initializer = tf.contrib.layers.xavier_initializer()),
            "bd1" : tf.get_variable('B5' , shape = (300) , initializer = tf.contrib.layers.xavier_initializer()),
            "bd2" : tf.get_variable('B6' , shape = (10) , initializer = tf.contrib.layers.xavier_initializer()) }

conv1 = tf.nn.conv2d(x , weights["wc1"] , strides = [1,1,1,1] , padding = "SAME")
conv1 = tf.nn.bias_add(conv1 , biases["bc1"])
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1 , ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
conv2 = tf.nn.conv2d(conv1 , weights["wc2"] , strides = [1,1,1,1] , padding = "SAME")
conv2 = tf.nn.bias_add(conv2 , biases["bc1"])
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2 , ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
conv3 = tf.nn.conv2d(conv2 , weights["wc3"] , strides = [1,1,1,1] , padding = "SAME")
conv3 = tf.nn.bias_add(conv3 , biases["bc1"])
conv3 = tf.nn.relu(conv3)
conv3 = tf.nn.max_pool(conv3 , ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
conv4 = tf.nn.conv2d(conv3 , weights["wc4"] , strides = [1,1,1,1] , padding = "SAME")
conv4 = tf.nn.bias_add(conv4 , biases["bc1"])
conv4 = tf.nn.relu(conv4)
conv4 = tf.nn.max_pool(conv4 , ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
fc1 = tf.reshape(conv4 , [-1 , weights["wd1"].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1 , weights["wd1"]), biases["bd1"])
fc1 = tf.nn.relu(fc1)
output = tf.add(tf.matmul(fc1 , weights["wd2"]) , biases["bd2"])
output = tf.nn.softmax(output)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output , labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        for batch in range(len(X_train)//batch_size):
            batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
            opt,loss = sess.run([optimizer,loss] , feed_dict = {x : batch_x , y : batch_y})
        if i%2 == 0:
            print("loss at iteration" + str(i) + "is {:.4f}".format(loss))
