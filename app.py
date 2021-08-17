import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from models import rfclassifier
from os import path

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    pass


@app.route('/uploader', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        f = request.files['file']
        csv_data = np.genfromtxt(f, delimiter=',')

        if not path.exists("./models/rfclassifer.joblib"):
            print("Model not found, creating new one")
            model_predictor = rfclassifier.PredictorModel()
            model_predictor.save_model()
            # as we are only predicting a single case, the data must be reshaped
            prediction = model_predictor.predict(csv_data.reshape(1, -1))

        else:
            print("Loading saved model")
            model_predictor = rfclassifier.PredictorModel(model='./models/rfclassifer.joblib')
            prediction = model_predictor.predict(csv_data.reshape(1, -1))

        pred_int = int(prediction)
        pred_string = ""
        if pred_int == 1:
            pred_string = "Low Risk"
        elif pred_int == 2:
            pred_string = "Medium Risk"
        elif pred_int == 3:
            pred_string = "High Risk"

        return render_template('predictor.html', data=f'This case is considered {pred_string}')

    elif request.method == 'GET':
        return render_template('predictor.html')


if __name__ == '__main__':
    app.run()
