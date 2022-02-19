from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

my_model = load('/predict_test/my_python_dir/logistic.pkl')

def my_prediction(arg):
    dummy = np.array(arg)
    dummyT = dummy.reshape(1,-1)
    a = dummy.shape
    b = dummyT.shape
    a_str = json.dumps(a)
    b_str = json.dumps(b)
    pred = my_model.predict(dummyT)
    pred = pred.tolist()
    pred = json.dumps(pred)
    return pred

