import numpy as np
from sklearn.metrics import mean_squared_error
"""
Used to measure error between actual and predicted values
"""
def rmse(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def mape(x1, x2, axis=0):
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)

    return np.mean(abs((x1-x2)/x1),axis=axis)*100


def smape(y_test,y_pred):

    y_test= np.asanyarray(y_test)
    y_pred=np.asanyarray(y_pred)

    return 1/len(y_test) * np.sum(2*np.abs(y_pred-y_test)/(np.abs(y_test)+np.abs(y_pred))*100)

def MAE(y_test, y_pred):
    mae = np.mean(np.abs(y_test-y_pred))
    return mae

def r2(y_test, y_pred):
    # to array
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    # count mean
    mean_y_true = np.mean(y_test)
    # count sse
    sse = np.sum((y_test-y_pred)**2)
    # count sst
    sst = np.sum((y_test-mean_y_true)**2)
    r2 = 1-(sse/sst)
    return r2