from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
from models import rfclassifier
from os import path

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/data', methods=['GET', 'POST'])
def dashboard():
    pass


@app.route('/uploader', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        f = request.files['file']
        df = pd.read_csv(f)
        x = df.iloc[0].values[:-1]

        if not path.exists("./models/rfclassifer.joblib"):
            print("Model not found, creating new one")
            model_predictor = rfclassifier.PredictorModel()
            model_predictor.save_model()
            prediction = model_predictor.predict(x.reshape(1, -1))

        else:
            print("Loading saved model")
            model_predictor = rfclassifier.PredictorModel(model='./models/rfclassifer.joblib')
            prediction = model_predictor.predict(x.reshape(1, -1))

        return render_template('predictor.html', data=prediction)

    elif request.method == 'GET':
        return render_template('predictor.html')


if __name__ == '__main__':
    app.run()
