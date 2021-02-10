'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from cuml.test.utils import array_equal
from cuml.utils.import_utils import has_xgboost
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cuml import ForestInference
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
import xgboost as xgb
import time

def train_xgboost_model(X_train, y_train,
                        num_rounds, model_path):
    # set the xgboost model parameters
    params = {'silent': 1, 'eval_metric':'error',
              'objective':'binary:logistic',
              'max_depth': 32}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # train the xgboost model
    bst = xgb.train(params, dtrain, num_rounds)

    # save the trained xgboost model
    bst.save_model(model_path)

    return bst



def predict_xgboost_model(X_validation, y_validation, xgb_model):

    # predict using the xgboost model
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    xgb_preds = xgb_model.predict(dvalidation)

    # convert the predicted values from xgboost into class labels
    xgb_preds = np.around(xgb_preds)
    return xgb_preds




# enter path to the directory where the trained model will be saved
model_path = './models/xgb.model'

# num of iterationfor which the model is trained
num_rounds = 1

col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)]  # Assign column names
dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)]           # Assign dtypes to each column
data = cudf.read_csv('HIGGS.csv', names=col_names, dtype=dtypes_ls)

X = data.iloc[:, 1:30].astype(np.float32) # Get data columns.  Must be float32 for our Classifier
y = data.iloc[:, 0 ].astype('category').cat.codes # Get labels column.  Will convert to int32

train_size = 0.8

# convert the dataset to np.float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# split the dataset into training and validation splits
X_train, X_validation, y_train, y_validation = cu_train_test_split(X, y, test_size = 1000000)

xgboost_model = train_xgboost_model(X_train, y_train, num_rounds, model_path)

start_time = time.time()
trained_model_preds = predict_xgboost_model(X_validation,
                                            y_validation,
                                            xgboost_model)


fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')


start_time = time.time()
fil_preds = fm.predict(X_validation)
end = (time.time() - start_time)*1000
print(end)

print("The shape of predictions obtained from xgboost : ",(trained_model_preds).shape)
print("The shape of predictions obtained from FIL : ",(fil_preds).shape)
print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_pre       ds, fil_preds))

'''
#################################################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from cuml.test.utils import array_equal
from cuml.utils.import_utils import has_xgboost
from cuml import ForestInference
from cuml.preprocessing.model_selection import train_test_split as cu_train_test_split
import time
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBRFClassifier
from numpy import asarray
import pandas as pd


'''col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)]  # Assign column names
dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)]           # Assign dtypes to each column
data = cudf.read_csv('HIGGS.csv', names=col_names, dtype=dtypes_ls)

X = data.iloc[:, 1:30].astype(np.float32) # Get data columns.  Must be float32 for our Classifier
y = data.iloc[:, 0 ].astype('category').cat.codes # Get labels column.  Will convert to int32'''


X_train = pd.read_csv("HIGGS.csv", nrows = 1000, header = None) # using 10M as training set
y_train = np.int8(X_train[0])
X_train = np.asarray(X_train.drop([0], axis = 1))

X_test = pd.read_csv("HIGGS.csv", skiprows =  10999900, header = None) # using 1M as testing set
y_test = np.int8(X_test[0])
X_test = np.asarray(X_test.drop([0], axis=1))

X_test = X_test.astype(np.float32)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)


# evaluate xgboost random forest algorithm for classification

# define the model
model = XGBRFClassifier(n_estimators=1, subsample=0.9, colsample_bynode=0.2)
print('fitting...')
model.fit(X_train, y_train)
print('fitted !')
trained_model_preds = model.predict(X_test)

model_path = './models/xgb.model'

model.save_model(model_path)

fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')

start_time = time.time()
fil_preds = fm.predict(X_test)
end = (time.time() - start_time)*1000
print(end)

print("The shape of predictions obtained from xgboost : ",(trained_model_preds).shape)
print("The shape of predictions obtained from FIL : ",(fil_preds).shape)
print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))


