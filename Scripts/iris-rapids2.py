import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cudf
from cuml.ensemble import RandomForestClassifier as RandomForestClassifier
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
import numpy as np
import time
# data link: https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv

# Read data
df = cudf.read_csv('../datasets/iris.csv', header = 0, delimiter = ',') # Get complete CSV

# Prep data
X = df.iloc[:, [0, 1, 2, 3]].astype(np.float32) # Get data columns.  Must be float32 for our Classifier
y = df.iloc[:, 4].astype('category').cat.codes # Get labels column.  Will convert to int32

train_data, test_data, train_label, test_label = cu_train_test_split(X, y, train_size=0.8)

while (len(test_data)<10000000):
    test_data = cudf.concat([test_data, test_data])

test1 = test_data[:1]
test100 = test_data[:100]
test1K = test_data[:1000]
test10K = test_data[:10000]
test100K = test_data[:100000]
test1M = test_data[:1000000]
test10M = test_data[:10000000]

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 1)
cu_s_random_forest.fit(train_data,train_label)


print('1 tree')
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications

start_time = time.time()
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test100K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 32)
cu_s_random_forest.fit(train_data,train_label)

print('1 tree')
predict = cu_s_random_forest.predict(test1,predict_model="GPU") # use GPU to do multi-class classifications


start_time = time.time()
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test10M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest


################################################################################################################
cu_s_random_forest = RandomForestClassifier(n_estimators = 64)
cu_s_random_forest.fit(train_data,train_label)

print('1 tree')
predict = cu_s_random_forest.predict(test1,predict_model="GPU") # use GPU to do multi-class classifications


start_time = time.time()
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test100K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test10M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################
cu_s_random_forest = RandomForestClassifier(n_estimators = 128)
cu_s_random_forest.fit(train_data,train_label)

print('1 tree')
predict = cu_s_random_forest.predict(test1,predict_model="GPU") # use GPU to do multi-class classifications


start_time = time.time()
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test10M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################
cu_s_random_forest = RandomForestClassifier(n_estimators = 256)
cu_s_random_forest.fit(train_data,train_label)

print('1 tree')
predict = cu_s_random_forest.predict(test1,predict_model="GPU") # use GPU to do multi-class classifications


start_time = time.time()
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test100K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test10M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest

################################################################################################################

cu_s_random_forest = RandomForestClassifier(n_estimators = 512)
cu_s_random_forest.fit(train_data,train_label)

print('1 tree')
predict = cu_s_random_forest.predict(test1,predict_model="GPU") # use GPU to do multi-class classifications


start_time = time.time()
predict = cu_s_random_forest.predict(test1, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test100, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test10K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)


start_time = time.time()
predict = cu_s_random_forest.predict(test100K, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test1M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

start_time = time.time()
predict = cu_s_random_forest.predict(test10M, predict_model="GPU") # use GPU to do multi-class classifications
print(time.time()-start_time)

del cu_s_random_forest
