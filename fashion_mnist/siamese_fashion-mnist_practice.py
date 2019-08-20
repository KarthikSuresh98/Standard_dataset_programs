import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input , Conv2D , MaxPooling2D , BatchNormalization , Flatten , Dense , Dropout , Concatenate , Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread_collection

loc = r'/home/karthik/Downloads/datasets/fashion_mnist/train.csv'
df = pd.read_csv(loc)
y = df['label']
del df['label']
X = np.asarray(df)
X = np.reshape(X , (-1,28,28,1)).astype('float32')/255.0
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 0)

train_groups = [X_train[np.where(y_train == i)] for i in np.unique(y_train)]
test_groups = [X_test[np.where(y_test == i)] for i in np.unique(y_test)]

def random_pairs(groups , size):
    out_img_a , out_img_b , out_score = [] , [] , []
    all_groups = list(range(len(groups)))
    for match_group in [True , False]:
        group_idx = np.random.choice(all_groups , size = size)
        out_img_a += [groups[c_idx][np.random.choice(range(groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
            out_score += [1]*size
        else:
            non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
            b_group_idx = non_group_idx
            out_score += [0]*size
        out_img_b += [groups[c_idx][np.random.choice(range(groups[c_idx].shape[0]))] for c_idx in b_group_idx]  
    return np.stack(out_img_a , 0) , np.stack(out_img_b , 0) , np.stack(out_score , 0)  


def plot(t_groups , n_examples):
    pv_a, pv_b, pv_sim = random_pairs(t_groups, n_examples)
    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))
    for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):
        ax1.imshow(c_a[:,:,0])
        ax1.set_title('Image A')
        ax1.axis('off')
        ax2.imshow(c_b[:,:,0])
        ax2.set_title('Image B\n Similarity: %3.0f%%' % (100*c_d))
        ax2.axis('off')
    plt.show()



shape = X_train[0].shape
inp = Input(shape = shape)
out = inp
for i in range(2):
    out = Conv2D(8*2**i , (3,3) , activation = 'linear')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(16*2**i , (3,3) , activation = 'linear')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(pool_size = (2,2))(out)
out = Flatten()(out)
out = Dense(32 , activation = 'linear')(out)
out = Dropout(0.5)(out)
out = BatchNormalization()(out)
out = Activation('relu')(out)
model = Model(inputs = inp , outputs = out)

inp1 = Input(shape = shape)
inp2 = Input(shape = shape)
out1 = model(inp1)
out2 = model(inp2)
res = Concatenate()([out1 , out2])
res = Dense(16 , activation = 'linear')(res)
res = BatchNormalization()(res)
res = Activation('relu')(res)
res = Dense(4 , activation = 'linear')(res)
res = BatchNormalization()(res)
res = Activation('relu')(res)
res = Dense(1 , activation = 'sigmoid')(res)

siamese_model = Model(inputs = [inp1 , inp2] , outputs = res)
siamese_model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['mae'])




def show_model_output(nb_examples = 3):
    pv_a, pv_b, pv_sim = random_pairs(test_groups, nb_examples)
    pred_sim = siamese_model.predict([pv_a, pv_b])
    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, pred_sim, m_axs.T):
        ax1.imshow(c_a[:,:,0])
        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*c_d))
        ax1.axis('off')
        ax2.imshow(c_b[:,:,0])
        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p_d))
        ax2.axis('off')
    plt.show() 


def siam_gen(groups , batch_size = 32):
    while True:
        pv_a , pv_b , pv_sim = random_pairs(groups , batch_size//2)
        yield [pv_a , pv_b] , pv_sim

valid_a , valid_b , valid_sim = random_pairs(test_groups , 1024)
loss_history = siamese_model.fit_generator(siam_gen(train_groups) , steps_per_epoch = 500 , validation_data = ([valid_a , valid_b] , valid_sim) , epochs = 10 , verbose = True)

_ = show_model_output()







