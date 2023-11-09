from flask import Flask, request, render_template
from flask_cors import cross_origin

from Pipeline.Pipeline01Preprocessing.dataReadingAndCleaningFile import dataReadingAndCleaningClass
from Pipeline.Pipeline03Prediction.predictionDataPreprocessingFile import predictionDataPreprocessingClass
from Pipeline.Pipeline03Prediction.predictionFile import predictionClass
from Initiator03Prediction import predictionInitiatorClass

app = Flask(__name__)


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method=="GET":
        return render_template(home.html)
    else:

        Airline = request.form.get('Airline'),
        Date_of_Journey = request.form.get('Dep_Time'),
        Source = request.form.get('Source'),
        Destination = request.form.get('Destination'),
        Dep_Time = request.form.get('Dep_Time'),
        Arrival_Time = request.form.get('Arrival_Time'),
        Duration = request.form.get('Duration'),
        Total_Stops = request.form.get('Total_Stops')

        predictionInitiatorObj =  predictionInitiatorClass()
        output = predictionInitiatorObj.receiveDataFromUI(Airline=Airline,Date_of_Journey=Date_of_Journey,Source=Source,
                                                          Destination=Destination,Dep_Time=Dep_Time,Arrival_Time=Arrival_Time,
                                                          Duration=Duration,Total_Stops=Total_Stops)

        #pred_df.to_csv("D:\\MachineLearningProjects\\PROJECT\\Airline-4\\cleanedData\\abc.csv")

        return render_template('home.html', prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
