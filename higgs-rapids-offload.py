import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.model_selection import train_test_split
import cudf
import time
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
from cuml import ForestInference

col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column
print('reading data, turning to cuDF')
data = cudf.read_csv('HIGGS.csv', names=col_names, dtype=dtypes_ls)



###################################################################################################################



X = data.iloc[:, 1:30].astype(np.float32) # Get data columns.  Must be float32 for our Classifier
y = data.iloc[:, 0 ].astype('category').cat.codes # Get labels column.  Will convert to int32

X_train, X_test, y_train, y_test = cu_train_test_split(X, y, test_size = 1000000)


#X, y = data[data.columns.difference(['label'])].as_matrix(), data['label'].to_array() # Separate data into X and y
#del data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000000)

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



###########################################################################################################

print('trees: 1')

cu_rf_params = {
            'n_estimators': 1,
            }


cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)



###########################################################################################################

print('trees: 32')

cu_rf_params = {
            'n_estimators': 32 ,
            }


#cu_rf = cuRF(**cu_rf_params)
#cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')

end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)



###########################################################################################################

print('trees: 64')

cu_rf_params = {
            'n_estimators': 64,
            }


#cu_rf = cuRF(**cu_rf_params)
#cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)



###########################################################################################################

print('trees: 128')

cu_rf_params = {
            'n_estimators': 128,
            }


#cu_rf = cuRF(**cu_rf_params)
#cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)



###########################################################################################################

print('trees: 256')

cu_rf_params = {
            'n_estimators': 256,
            }

#cu_rf = cuRF(**cu_rf_params)
#cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
#pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)



###########################################################################################################

print('trees: 512')

cu_rf_params = {
            'n_estimators': 512,
            }


cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
pred = cu_rf.predict(cudf.read_csv('./datasets/higgs10M.csv').iloc[:,1:30].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print(end)



