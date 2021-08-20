import pandas as pd
from flask import render_template
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from models import rfclassifier
from sklearn.metrics import confusion_matrix
import numpy as np

class Plotter:
    def __init__(self):
        self.df = pd.read_csv('./data/fetal_health.csv')

    def plot_targets(self):
        plt.figure(figsize=(4, 3))
        sns.set_theme(style="darkgrid", palette="coolwarm")
        ax = sns.countplot(x=self.df['fetal_health'])
        ax.set_xticklabels(('Low Risk', 'Medium Risk', 'High Risk'))
        ax.set(xlabel="Fetal Health", ylabel="Count")
        plt.title("Count of Health Targets")
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def plot_heatmap(self):
        plt.figure(figsize=(14, 12))
        plt.title("Correlation Map")
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", linewidth=1)

        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def correlation_plot(self):
        over_50 = self.df[self.df["percentage_of_time_with_abnormal_long_term_variability"] > 50.0]
        sns.scatterplot(data=over_50, x="abnormal_short_term_variability",
                        y="percentage_of_time_with_abnormal_long_term_variability",
                        palette="bright",
                        hue="fetal_health")
        plt.axhline(over_50["percentage_of_time_with_abnormal_long_term_variability"].mean(), color="red",
                    linestyle="--")
        plt.ylabel("% of Time With Long Term Abnormal Variability")
        plt.xlabel("Abnormal Short Term Variability")
        plt.title("Fetal Heart Rate (FHR) Variability Chart")

        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def pie_plot(self):
        plt.figure(figsize=(4, 2))
        pie_plot = plt.pie(self.df['fetal_health'].value_counts(), labels=["Low Risk", "Medium Risk", "High Risk"],
                           autopct='%1.f%%')
        plt.title("Fetal Health Counts")
        plt.xlabel("Fetal Health")
        plt.ylabel("Cases")

        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return plot_url

    def confusion_matrix(self):
        model = rfclassifier.PredictorModel()
        plt.subplots(figsize=(6, 5))
        prediction = model.predict(model.X_test)
        matrix = confusion_matrix(model.y_test, prediction)
        sns.heatmap(matrix/np.sum(matrix), cmap="coolwarm", annot=True)
        plt.title("Model Accuracy")
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")

        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return plot_url
