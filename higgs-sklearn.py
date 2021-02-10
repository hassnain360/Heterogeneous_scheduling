import pandas as pd
import numpy as np
import sklearn
import time
import numpy
import pickle
from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv("HIGGS.csv", nrows = 10000000, header = None)
y_train = np.int8(X_train[0])
X_train = np.asarray(X_train.drop([0], axis = 1))

X_test = pd.read_csv("HIGGS.csv", skiprows =  10000000, header = None)
y_test = np.int8(X_test[0])
X_test = np.asarray(X_test.drop([0], axis=1))

########################################################################################################
print('Training Random Forest with HIGGS - 1 Forest ...')
RF_HIGGS_1 = RandomForestClassifier(n_estimators=1)
RF_HIGGS_1.fit(X_train, y_train)
filename='./models/RF_HIGGS_SK_1.sav'
pickle.dump(RF_HIGGS_1,open(filename,'wb+'))
print('Scoring Random Forest with HIGGS - 1 Forest ...')


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1])
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
RF_HIGGS_1.predict(X_test[:100])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:10000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:100000])
end = (time.time() - start_time)*1000
print(end)



start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000000])
end = (time.time() - start_time)*1000
print(end)

########################################################################################################
print('Training Random Forest with HIGGS - 32 Forest ...')
RF_HIGGS_1 = RandomForestClassifier(n_estimators=32)
RF_HIGGS_1.fit(X_train, y_train)
filename='./models/RF_HIGGS_SK_32.sav'
pickle.dump(RF_HIGGS_1,open(filename,'wb+'))



start_time = time.time()
RF_HIGGS_1.predict(X_test[:1])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:100])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:10000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:100000])
end = (time.time() - start_time)*1000
print(end)



start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000000])
end = (time.time() - start_time)*1000
print(end)

########################################################################################################
print('Training Random Forest with HIGGS - 64 Forest ...')
RF_HIGGS_1 = RandomForestClassifier(n_estimators=64)
RF_HIGGS_1.fit(X_train, y_train)
filename='./models/RF_HIGGS_SK_64.sav'
pickle.dump(RF_HIGGS_1,open(filename,'wb+'))

print('Scoring Random Forest with HIGGS - 1 Forest ...')


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:100])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:10000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:100000])
end = (time.time() - start_time)*1000
print(end)



start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000000])
end = (time.time() - start_time)*1000
print(end)

########################################################################################################
print('Training Random Forest with HIGGS - 1 Forest ...')
RF_HIGGS_1 = RandomForestClassifier(n_estimators=128)
RF_HIGGS_1.fit(X_train, y_train)
filename='./models/RF_HIGGS_SK_128.sav'
pickle.dump(RF_HIGGS_1,open(filename,'wb+'))

print('Scoring Random Forest with HIGGS - 1 Forest ...')


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:100])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000])
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
RF_HIGGS_1.predict(X_test[:10000])
end = (time.time() - start_time)*1000
print(end)

start_time = time.time()
RF_HIGGS_1.predict(X_test[:100000])
end = (time.time() - start_time)*1000
print(end)


start_time = time.time()
RF_HIGGS_1.predict(X_test[:1000000])
end = (time.time() - start_time)*1000
print(end)

