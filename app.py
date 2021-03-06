import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from models import rfclassifier

import plots

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    plotter = plots.Plotter()
    target_data = plotter.plot_targets()
    heatmap = plotter.plot_heatmap()
    pie_plot = plotter.pie_plot()
    conf_matrix = plotter.confusion_matrix()
    classification_report = plotter.class_report_table()

    return render_template('dashboard.html', target_data=target_data, heatmap=heatmap,
                           pie_plot=pie_plot, conf_matrix=conf_matrix,
                           classification_table=classification_report.to_html())


@app.route('/uploader', methods=['GET', 'POST'])
def predict_csv():
    if request.method == 'POST':
        f = request.files['file']
        csv_data = np.genfromtxt(f, delimiter=',')

        model_predictor = rfclassifier.PredictorModel()
        # as we are only predicting a single case, the data must be reshaped
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
