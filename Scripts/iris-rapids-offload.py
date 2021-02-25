import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cudf
from cuml.ensemble import RandomForestClassifier as RandomForestClassifier
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
import numpy as np
import time
# data link: https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv

# Read data
df = cudf.read_csv('./iris.csv', header = 0, delimiter = ',') # Get complete CSV

# Prep data
X = df.iloc[:, [0, 1, 2, 3]].astype(np.float32) # Get data columns.  Must be float32 for our Classifier
y = df.iloc[:, 4].astype('category').cat.codes # Get labels column.  Will convert to int32

train_data, test_data, train_label, test_label = cu_train_test_split(X, y, train_size=0.8)

while (len(test_data)<10000000):
    test_data = cudf.concat([test_data, test_data])


test_data[:1].to_csv('./datasets/iris1.csv')
test_data[:100].to_csv('./datasets/iris100.csv')
test_data[:1000].to_csv('./datasets/iris1K.csv')
test_data[:10000].to_csv('./datasets/iris10K.csv')
test_data[:100000].to_csv('./datasets/iris100K.csv')
test_data[:1000000].to_csv('./datasets/iris1M.csv')
test_data[:10000000].to_csv('./datasets/iris10M.csv')


################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 1)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1K.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1M.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10M.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 32)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1K.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1M.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10M.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 64)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1K.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1M.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10M.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 128)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1K.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1M.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10M.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 256)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1K.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1M.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10M.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 512)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1K.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris100K.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris1M.csv').iloc[:,1:5].astype('float32') ,predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(cudf.read_csv('./datasets/iris10M.csv').iloc[:,1:5].astype('float32'), predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

