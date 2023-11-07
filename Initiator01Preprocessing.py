from ExceptionLoggerAndUtils.exception import CustomException
from ExceptionLoggerAndUtils.logger import App_Logger
from Pipeline.Pipeline01Preprocessing.dataReadingAndCleaningFile import dataReadingAndCleaningClass
import os
import sys
import pandas as pd

class preprocessingInitiatorClass():
    def __init__(self):

        self.cwd=os.getcwd()
        self.file_object = open(self.cwd+'preprocessing.txt', 'a+')


        self.log_writer = App_Logger()
        self.preprocessingObj = dataReadingAndCleaningClass()
        self.File_Path = "C:\\Users\\shind\\JupiterWorking\\iNuron\\EDA\\Data Travel\\Data_Train.xlsx"




    def preprocessingInitiatorMethod(self):
        try:

            #self.log_writer(self.file_object, "Starting preprocessing Steps!!!")


            column_names, NumberofColumns, airlineName = self.preprocessingObj.valuesFromSchema()
            df = self.preprocessingObj.readingDataSet(self.File_Path)
            df = self.preprocessingObj.removeNullValues(df)
            df = self.preprocessingObj.removingUnevenValues(df)
            df = self.preprocessingObj.removingOutlier(column_names,airlineName,df)
            df = self.preprocessingObj.convertDateInToDayMonthYear(df)
            df = self.preprocessingObj.convertHoursAndMinutesToIndependantColumns(df=df,columName="Dep_Time")
            df = self.preprocessingObj.convertHoursAndMinutesToIndependantColumns(df=df,columName="Arrival_Time")
            df = self.preprocessingObj.convertDurationToMunutes(df)
            df = self.preprocessingObj.dropUncessaryColumns(df)
            self.preprocessingObj.saveDataToFolder(df)

            print(df)

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    preproInitiatorObj = preprocessingInitiatorClass()
    preproInitiatorObj.preprocessingInitiatorMethod()
