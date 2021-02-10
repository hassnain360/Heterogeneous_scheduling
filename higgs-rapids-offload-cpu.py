import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # ONLY USE 1 GPU
from cuml.ensemble import RandomForestClassifier as cuRF
import cudf
import time
import numpy as np
import pandas as pd
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
#from cuml import ForestInference

col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)]  # Assign column names
dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)]           # Assign dtypes to each column
data = cudf.read_csv('HIGGS.csv', names=col_names, dtype=dtypes_ls)

X = data.iloc[:, 1:30].astype(np.float32) # Get data columns.  Must be float32 for our Classifier
y = data.iloc[:, 0 ].astype('category').cat.codes # Get labels column.  Will convert to int32

X_train, X_test, y_train, y_test = cu_train_test_split(X, y, test_size = 1000000)

'''
while (len(X_test)<10000000):
    X_test=cudf.concat([X_test,X_test])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_test[:1].to_csv('./datasets/higgs1.csv')
X_test[:100].to_csv('./datasets/higgs100.csv')
X_test[:1000].to_csv('./datasets/higgs1K.csv')
X_test[:10000].to_csv('./datasets/higgs10K.csv')
X_test[:100000].to_csv('./datasets/higgs100K.csv')
X_test[:1000000].to_csv('./datasets/higgs1M.csv')
X_test[:10000000].to_csv('./datasets/higgs10M.csv')
'''

##########################################################################################################
''' INFERENCE  + OFFLOADING '''
###########################################################################################################


print('trees: 1')

cu_rf_params = {
            'n_estimators': 1,
            'max_depth': -1,
            }


cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

#############################################################################################################

print('trees: 32')

cu_rf_params = {
            'n_estimators': 32,
            'max_depth': -1,
            }


cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)


#############################################################################################################

print('trees: 64')

cu_rf_params = {
            'n_estimators': 64,
            'max_depth': -1,
            }


cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

#############################################################################################################

print('trees: 128')

cu_rf_params = {
            'n_estimators': 128,
            'max_depth': -1,
            }


cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='CPU')
end = (time.time() - start_time)*1000
print(end)
