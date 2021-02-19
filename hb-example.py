
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from hummingbird.ml import convert


# We are going to use the breast cancer dataset from scikit-learn for this example.
X, y = load_breast_cancer(return_X_y=True)
nrows=15000
X = X[0:nrows]
y = y[0:nrows]


# Create and train a random forest model.
model = RandomForestClassifier(n_estimators=10, max_depth=10)
model.fit(X, y)

# Create and train a random forest model.
model = RandomForestClassifier(n_estimators=10, max_depth=10)
model.fit(X, y)

%%timeit -r 3
# Time for scikit-learn.
model.predict(X)
model = convert(model, 'torch', extra_config={"tree_implementation":"gemm"})

%%timeit -r 3
# Time for HB.
model.predict(X)
model.to('cuda')
PyTorchBackendModelClassification(
  (operator_map): ModuleDict(
    (SklearnRandomForestClassifier): GEMMDecisionTreeImpl()
  )
)

%%timeit -r 3
# Time for HB GPU.
model.predict(X)