import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from models import rfclassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class Plotter:
    def __init__(self):
        self.df = pd.read_csv('./data/fetal_health.csv')
        self.model = rfclassifier.PredictorModel()

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
        heatmap = sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", linewidth=1)
        self.format_heatmap(heatmap)
        heatmap.figure.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def format_heatmap(self, heatmap):
        ax = heatmap
        new_xlabels = []
        new_ylabels = []
        xtick_labels = [x.get_text() for x in ax.get_xticklabels()]
        ytick_labels = [y.get_text() for y in ax.get_yticklabels()]
        for x in xtick_labels:
            new_xlabels.append(x.replace("_", " ").title())

        for y in ytick_labels:
            new_ylabels.append(y.replace("_", " ").title())

        ax.set_xticklabels(new_xlabels)
        ax.set_yticklabels(new_ylabels)


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
        plt.figure()
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
        model = self.model
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

    def classification_report(self):
        pred = self.model.predict(self.model.X_test)
        class_report = classification_report(self.model.y_test, pred, output_dict=True)
        df_classification_report = pd.DataFrame(class_report).transpose()
        df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
        return df_classification_report

    def class_report_table(self):
        data = self.classification_report()
        return data
