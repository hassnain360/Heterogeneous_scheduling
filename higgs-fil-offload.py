import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from cuml.test.utils import array_equal
from cuml import ForestInference
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
import time
from xgboost import XGBRFClassifier
from numpy import asarray
import pandas as pd

#X_train = pd.read_csv("HIGGS.csv", nrows = 10000000, header = None) # using 10M as training set
#y_train = np.int8(X_train[0])
#X_train = np.asarray(X_train.drop([0], axis = 1))

X_test = pd.read_csv("HIGGS.csv", skiprows =  10000000, header = None) # using 1M as testing set
y_test = np.int8(X_test[0])
X_test = np.asarray(X_test.drop([0], axis=1))

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)
#X_train = X_train.astype(np.float32)
#y_train = y_train.astype(np.float32)


#################################################################################################

#model = XGBRFClassifier(n_estimators=1, subsample=0.9, colsample_bynode=0.2)
print("1 Tree.")
#model.fit(X_train, y_train)
model_path = './models/xgb1'
model.save_model(model_path)
fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

#print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))


#################################################################################################

#model = XGBRFClassifier(n_estimators=32, subsample=0.9, colsample_bynode=0.2)
print("32 Trees")
#model.fit(X_train, y_train)
model_path = './models/xgb32'
model.save_model(model_path)
fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

#print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))

#################################################################################################

#model = XGBRFClassifier(n_estimators=64, subsample=0.9, colsample_bynode=0.2)
print("64 Trees")
#model.fit(X_train, y_train)
model_path = './models/xgb64'
model.save_model(model_path)
fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

#print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))

#################################################################################################

#model = XGBRFClassifier(n_estimators=128, subsample=0.9, colsample_bynode=0.2)
print("128 Trees")
#model.fit(X_train, y_train)
model_path = './models/xgb128'
model.save_model(model_path)
fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs10K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs100K.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
fil_preds = fm.predict(cudf.read_csv('./datasets/higgs1M.csv').iloc[:,1:30].astype('float32'))
end = (time.time() - start_time)*1000
print(end)

#print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))


