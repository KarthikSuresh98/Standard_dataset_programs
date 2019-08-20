import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

loc = r'/home/karthik/Downloads/parkinsons_updrs.csv'
df = pd.read_csv(loc)
del df['subject#']
output1 = np.asarray(df['motor_UPDRS'] , dtype = np.float32)
output2 = np.asarray(df['total_UPDRS'] , dtype = np.float32)
del df['motor_UPDRS']
del df['total_UPDRS']
data  = np.asarray(df , dtype = np.float32)
X_train , X_test , y_train , y_test = train_test_split(data , output1 , test_size = 0.1, random_state = 0)

clf = RandomForestRegressor(max_depth = 15 , min_samples_split = 6).fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
