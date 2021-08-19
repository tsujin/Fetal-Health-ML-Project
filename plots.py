import pandas as pd
from flask import render_template
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

class Plotter:
    def __init__(self):
        self.df = pd.read_csv('./data/fetal_health.csv')

    def plot_targets(self):
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
