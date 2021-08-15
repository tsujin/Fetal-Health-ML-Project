import csv

from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
from models import rfclassifier
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/data', methods=['GET', 'POST'])
def dashboard():
    return str(rfclassifier.test_model())


@app.route('/uploader', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        model_predictor = rfclassifier.PredictorModel()
        f = request.files['file']
        df = pd.read_csv(f)
        x = df.iloc[9].values[:-1]
        prediction = model_predictor.predict(x.reshape(1, -1))

        return render_template('predictor.html', data=prediction)

    elif request.method == 'GET':
        return render_template('predictor.html')


if __name__ == '__main__':
    app.run()
