import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # ONLY USE 1 GPU
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.model_selection import train_test_split
import cudf
import time
import numpy as np
import pandas as pd
#from cuml import ForestInference

col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column
data = cudf.read_csv('HIGGS.csv', names=col_names, dtype=dtypes_ls)
data.head().to_pandas()
###################################################################################################################

X, y = data[data.columns.difference(['label'])].as_matrix(), data['label'].to_array() # Separate data into X and y
del data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000000)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

cu_rf_params = {
            'n_estimators': 256,
            }

print("Training cuML Random Forest with 1 Tree...")

cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
#pred = cu_rf.predict(X_test[:1].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:100].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:1000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1K, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:10000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10k, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:100000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100K, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:1000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1M, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:10000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10M, time taken is: --- %s ms ---" % end)



#######################################################################################################################v


cu_rf_params = {
        'n_estimators': 512,
            }

print("Training cuML Random Forest with 32 Trees...")

cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
#pred = cu_rf.predict(X_test[:1].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:100].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:1000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1K, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:10000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10k, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:100000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100K, time taken is: --- %s ms ---" % end)

start_time = time.time()
#pred = cu_rf.predict(X_test[:1000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1M, time taken is: --- %s ms ---" % end)


start_time = time.time()
pred = cu_rf.predict(X_test[:10000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10M, time taken is: --- %s ms ---" % end)


#######################################################################################################################v











cu_rf_params = {
            'n_estimators': 64,
            }

print("Training cuML Random Forest with 64 Tree...")

cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
pred = cu_rf.predict(X_test[:1].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:100].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:1000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1K, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:10000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10k, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:100000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100K, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:1000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1M, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:10000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10M, time taken is: --- %s ms ---" % end)


#######################################################################################################################v








cu_rf_params = {
            'n_estimators': 128,
            }

print("Training cuML Random Forest with 128 Tree...")

cu_rf = cuRF(**cu_rf_params)
cu_rf.fit(X_train, y_train)

print("Training Complete !")

print("Now predicting...")
start_time = time.time()
pred = cu_rf.predict(X_test[:1].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:100].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:1000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1K, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:10000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10k, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:100000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 100K, time taken is: --- %s ms ---" % end)

start_time = time.time()
pred = cu_rf.predict(X_test[:1000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 1M, time taken is: --- %s ms ---" % end)


start_time = time.time()
pred = cu_rf.predict(X_test[:10000000].astype('float32'), predict_model='GPU')
end = (time.time() - start_time)*1000
print("For HIGGS, 1 Forest, with Scoring Batch Size: 10M, time taken is: --- %s ms ---" % end)


#######################################################################################################################v

