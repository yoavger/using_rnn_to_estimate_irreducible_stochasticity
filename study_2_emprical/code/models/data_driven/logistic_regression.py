import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression

def preprocess_logistic_regression(data, lag):
    """
    this funcation preprocessing the data for logistic regression 
    Args:
        data: dataframe of behavior to fit  

    Returns:
        X: a matrix of the predictors variables
        y: a vector of the response variable 
        
    """
    
    df = data
    
    X = []
    for j in range(1 , lag+1):
        a = df['action_stage_1'].shift(j, fill_value=0).values
        a[a==0]=-1
        r = df['reward'].shift(j, fill_value=0).values
        r[r==0]=-1
        t = df['state_of_stage_2'].shift(j, fill_value=0).values
        t[t==0]=-1

        a_r = a*r
        a_t = a*t
        r_t = r*t
        a_r_t = a*r*t

        X.append(np.vstack((a, r, t,
                   a_r, a_t, r_t,
                   a_r_t)))

    y = df['action_stage_1'].values
    X = np.array(X).reshape(lag*7, len(df))
    X = X.T
    return X, y

def fit_logistic_regression(X,y):
    """
    fit logistic regression on the data
    
    Args:
        X: a matrix of the predictors variables
        y: a vector of the response  

    Returns:
        lm: the model aftre fit
        lm.intercept_: the intercept coefficient
        lm.coef_ : the coefficient of the other predictors
    """
    if np.all(y == y[0]):
        return None, None, None
    else:
        lm = LogisticRegression()
        lm.fit(X, y)
        return lm, lm.intercept_, lm.coef_ 