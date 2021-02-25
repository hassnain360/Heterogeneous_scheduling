import sklearn
import time
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

x_test=[]
y_test = list(y_test)
for i in range(1000000):
    x_test.append(X_test[i%38])
    if (i>=38):
        y_test.append(y_test[i%38])


##############################################################################################################################

print('Training Random Forest with IRIS - 1 Forest ...')
RF_IRIS_1 = RandomForestClassifier(n_estimators=1)
RF_IRIS_1.fit(X_train, y_train)
print('Scoring Random Forest with IRIS - 1 Forest ...')


start_time = time.time()
RF_IRIS_1.predict(x_test[:1])
print("For IRIS, 1 Forest, with Scoring Batch Size: 1, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_1.predict(x_test[:100])
print("For IRIS, 1 Forest, with Scoring Batch Size: 100, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_1.predict(x_test[:1000])
print("For IRIS, 1 Forest, with Scoring Batch Size: 1K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_1.predict(x_test[:10000])
print("For IRIS, 1 Forest, with Scoring Batch Size: 10K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_1.predict(x_test[:100000])
print("For IRIS, 1 Forest, with Scoring Batch Size: 100K, time taken is: --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
RF_IRIS_1.predict(x_test[:1000000])
print("For IRIS, 1 Forest, with Scoring Batch Size: 1M, time taken is: --- %s seconds ---" % (time.time() - start_time))



##############################################################################################################################

print('Training Random Forest with IRIS - 32 Forests ...')
RF_IRIS_32 = RandomForestClassifier(n_estimators=32)
RF_IRIS_32.fit(X_train, y_train)
print('Scoring Random Forest with IRIS - 32 Forest ...')



start_time = time.time()
RF_IRIS_32.predict(x_test[:1])
print("For IRIS, 32 Forests, with Scoring Batch Size: 1, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_32.predict(x_test[:100])
print("For IRIS, 32 Forests, with Scoring Batch Size: 100, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_32.predict(x_test[:1000])
print("For IRIS, 32 Forests, with Scoring Batch Size: 1K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_32.predict(x_test[:10000])
print("For IRIS, 32 Forests, with Scoring Batch Size: 10K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_32.predict(x_test[:100000])
print("For IRIS, 32 Forests, with Scoring Batch Size: 100K, time taken is: --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
RF_IRIS_32.predict(x_test[:1000000])
print("For IRIS, 32 Forests, with Scoring Batch Size: 1M, time taken is: --- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################

print('Training Random Forest with IRIS - 64 Forests ...')
RF_IRIS_64 = RandomForestClassifier(n_estimators=64)
RF_IRIS_64.fit(X_train, y_train)
print('Scoring Random Forest with IRIS - 64 Forest ...')


start_time = time.time()
RF_IRIS_64.predict(x_test[:1])
print("For IRIS, 64 Forests, with Scoring Batch Size: 1, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_64.predict(x_test[:100])
print("For IRIS, 64 Forests, with Scoring Batch Size: 100, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_64.predict(x_test[:1000])
print("For IRIS, 64 Forests, with Scoring Batch Size: 1K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_64.predict(x_test[:10000])
print("For IRIS, 64 Forests, with Scoring Batch Size: 10K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_64.predict(x_test[:100000])
print("For IRIS, 64 Forests, with Scoring Batch Size: 100K, time taken is: --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
RF_IRIS_64.predict(x_test[:1000000])
print("For IRIS, 64 Forests, with Scoring Batch Size: 1M, time taken is: --- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################
print('Training Random Forest with IRIS - 128 Forests ...')
RF_IRIS_128 = RandomForestClassifier(n_estimators=128)
RF_IRIS_128.fit(X_train, y_train)
print('Scoring Random Forest with IRIS - 128 Forest ...')


start_time = time.time()
RF_IRIS_128.predict(x_test[:1])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_128.predict(x_test[:100])
print("For IRIS, 128 Forests, with Scoring Batch Size: 100, time taken is: --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
RF_IRIS_128.predict(x_test[:1000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_128.predict(x_test[:10000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 10K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_128.predict(x_test[:100000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 100K, time taken is: --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
RF_IRIS_128.predict(x_test[:1000000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1M, time taken is: --- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################


##############################################################################################################################
print('Training Random Forest with IRIS - 256 Forests ...')
RF_IRIS_256 = RandomForestClassifier(n_estimators=256)
RF_IRIS_256.fit(X_train, y_train)
print('Scoring Random Forest with IRIS - 256 Forest ...')


start_time = time.time()
RF_IRIS_256.predict(x_test[:1])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_256.predict(x_test[:100])
print("For IRIS, 128 Forests, with Scoring Batch Size: 100, time taken is: --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
RF_IRIS_256.predict(x_test[:1000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_256.predict(x_test[:10000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 10K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_256.predict(x_test[:100000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 100K, time taken is: --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
RF_IRIS_256.predict(x_test[:1000000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1M, time taken is: --- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################
##############################################################################################################################
print('Training Random Forest with IRIS - 512 Forests ...')
RF_IRIS_512 = RandomForestClassifier(n_estimators=512)
RF_IRIS_512.fit(X_train, y_train)
print('Scoring Random Forest with IRIS - 128 Forest ...')


start_time = time.time()
RF_IRIS_512.predict(x_test[:1])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_512.predict(x_test[:100])
print("For IRIS, 128 Forests, with Scoring Batch Size: 100, time taken is: --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
RF_IRIS_512.predict(x_test[:1000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_512.predict(x_test[:10000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 10K, time taken is: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
RF_IRIS_512.predict(x_test[:100000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 100K, time taken is: --- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
RF_IRIS_512.predict(x_test[:1000000])
print("For IRIS, 128 Forests, with Scoring Batch Size: 1M, time taken is: --- %s seconds ---" % (time.time() - start_time))

##############################################################################################################################
