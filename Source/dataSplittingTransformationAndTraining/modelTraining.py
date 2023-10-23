import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
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
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Linear Regression": {},

                # #"XGBRegressor": {
                #     'learning_rate': [.1, .01, .05, .001],
                #     'n_estimators': [8, 16, 32, 64, 128, 256]
                # },

                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

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
