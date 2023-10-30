import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from ExceptionLoggerAndUtils.logger import App_Logger
from ExceptionLoggerAndUtils.exception import CustomException
#from xgboost import XGBRegressor

from ExceptionLoggerAndUtils.utils import save_object, evaluate_Regression_models


class ModelTrainerClass:
    def __init__(self):
        self.log_writer = App_Logger()
        self.trained_model_file_path = os.path.join("artifacts", "model.pkl")

    def modelsToTrainAndParameters(self):
        try:
            models = {
                "Random Forest"         : RandomForestRegressor(),
                "Gradient Boosting"     : GradientBoostingRegressor(),
                "Linear Regression"     : LinearRegression(),
                "lasso"                 : Lasso(),
                "ridge"                 : Ridge(),
                "AdaBoost Regressor"    : AdaBoostRegressor(),
                "SVR"                   : SVR()

            }

            params = {
                    "Random Forest": {
                        'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
                        'max_depth': [None, 10, 20, 30],   # Maximum depth of each tree
                        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
                        'min_samples_leaf': [1, 2, 4],    # Minimum samples required in a leaf node
                        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider
                    },
                        "Gradient Boosting": {
                        'n_estimators': [100, 200, 300],  # Number of boosting stages (trees)
                        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
                        'max_depth': [3, 4, 5],  # Maximum depth of each tree
                        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
                        'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
                        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider
                    },
                    "Linear Regression": {
                        #'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength (alpha)
                        'fit_intercept': [True, False]  # Whether to fit the intercept
                    },
                    "lasso": {
                        'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength (alpha)
                        'fit_intercept': [True, False]  # Whether to fit the intercept
                    },
                    "ridge": {
                        'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength (alpha)
                        'fit_intercept': [True, False]  # Whether to fit the intercept
                    },
                    "AdaBoost Regressor":{
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.01, 0.001],
                        'base_estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)],
                        'loss': ['linear', 'square', 'exponential']
                    },
                    "SVR":{
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    },
                }

            return models , params
        except Exception as e:
            raise CustomException(e, sys)


    def modelTraingMethod(self,X_train, X_test, y_train, y_test,models,params):
        try:

            model_report: dict = evaluate_Regression_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            ## below returns the best model score from dict
            best_model_score = max(sorted(model_report.values()))


            ## below returns the best model Name score from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            ## To get best model name from dict
            best_model = models[best_model_name]

            print("Best found model on both training and testing dataset")
            print(best_model)

            save_object(
                file_path=self.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
