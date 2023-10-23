import sys
import pandas as pd
from ExceptionLoggerAndUtils.logger import App_Logger
from ExceptionLoggerAndUtils.exception import CustomException
from ExceptionLoggerAndUtils.utils import load_object

class PredictPipeline():
    def __init__(self):
        pass

    def predict(self,features):
        try:
            modelPath = "artifacts/model.pkl"
            preprocessorPath = "artifacts/transformation.pkl"
            model = load_object(file_path=modelPath)
            transformation = load_object(file_path=preprocessorPath)
            dataScaled = transformation.transform(features)
            pred = model.predict(dataScaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)



class CustomData():
        def __init__(self,Airline:str,Date_of_Journey:str,Source:str,Destination:str,
                     Dep_Time:str,Arrival_Time:str,Duration:str,Total_Stops:str):
            self.Airline = Airline
            self.Date_of_Journey = Date_of_Journey
            self.Source = Source
            self.Destination = Destination
            self.Dep_Time = Dep_Time
            self.Arrival_Time = Arrival_Time
            self.Duration = Duration
            self.Total_Stops = Total_Stops

        def getDataAsDataFrame(self):
            inputDict = {
                "Airline": [self.Airline],
                "Date_of_Journey": [self.Date_of_Journey],
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Dep_Time": [self.Dep_Time],
                "Arrival_Time": [self.Arrival_Time],
                "Duration": [self.Duration],
                "Total_Stops": [self.Total_Stops]

            }

            return pd.DataFrame(inputDict)


