import os
import sys
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
#import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from ExceptionLoggerAndUtils.exception import CustomException


def save_object(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_Regression_models(X_train, y_train, X_test, y_test, models, param):
    try:
        ar2Score = {}
        j = 0
        k = 0
        print(j)

        for i in range(len(list(models))):
            print(j)
            j = j + 1
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            # model.fit(X_train, y_train)  # Train model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            R2test_model_score = r2_score(y_test, y_test_pred)
            aR2 = 1 - (1 - R2test_model_score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

            MSE = metrics.mean_squared_error(y_test, y_test_pred)
            MAE = metrics.mean_absolute_error(y_test, y_test_pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

            ar2Score[list(models.keys())[i]] = aR2

            modelName = list(models.keys())[i]
            modelScore = createDictOfScores(modelName, modelScore, R2test_model_score, aR2, MSE, MAE, RMSE)

            k = k+1

        print(modelScore)
        print(ar2Score)

        return ar2Score

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def createDictOfScores(modelName,modelScore,R2test_model_score,aR2,MSE,MAE,RMSE):
    modelScore = {}

    if modelName in modelScore:
        modelScore[modelName].append(R2test_model_score)
    else:
        modelScore[modelName] = R2test_model_score

    if modelName in modelScore:
        modelScore[modelName].append(aR2)
    else:
        modelScore[modelName] = aR2

    if modelName in modelScore:
        modelScore[modelName].append(MSE)
    else:
        modelScore[modelName] = MSE

    if modelName in modelScore:
        modelScore[modelName].append(MAE)
    else:
        modelScore[modelName] = MAE

    if modelName in modelScore:
        modelScore[modelName].append(RMSE)

    else:
        modelScore[modelName] = RMSE


    return modelScore