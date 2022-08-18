import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


def abstention_fit(model, x_test,y_test, low = 0, high = 0.16, step = 0.01, error = 'symmetric'):
    if error == 'symmetric':
        intervals = np.arange(low, high, step=step)
        mcc_values = np.zeros(len(intervals))
        sizes = np.zeros(len(intervals))

        for i in range(len(intervals)):
            mcc, size = cost(model, x_test, y_test, intervals[i])
            mcc_values[i] = mcc
            print(i,mcc)
        best = intervals[(mcc_values*sizes).argmax()]
        return(best)
        
    else:
        return None

def cost(model, x_test,y_test,interval):
    pred_probs = model.predict_proba(x_test)[:,1]
    #max_prob = np.max(pred_probs)
    mask = np.abs(pred_probs-0.5) > interval
    size = np.sum(mask)/len(mask)
    _x = x_test[mask]
    _y = y_test[mask]
    _y_hat = model.predict(_x)
    coef = matthews_corrcoef(_y,_y_hat)
    return(coef,size)
