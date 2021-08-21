from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from os import path


class PredictorModel:
    def __init__(self):
        # new model, we should make and train from scratch
        if not path.exists('./models/rfclassifier.joblib'):
            self.data = pd.read_csv("./data/fetal_health.csv")
            self.model = RandomForestClassifier(max_depth=20, random_state=42)

            self.X = self.data.drop('fetal_health', axis=1)
            self.y = self.data['fetal_health']

            # dealing with an imbalance in target values
            oversample = SMOTE()
            self.X, self.y = oversample.fit_resample(self.X, self.y)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                    self.y,
                                                                                    test_size=0.2)
            self.model.fit(self.X_train, self.y_train)

            self.save_model()

        # model exists, so load it
        else:
            from joblib import load
            self.model = load('./models/rfclassifier.joblib')

    # used to predict a single case
    def predict(self, data):
        return self.model.predict(data)

    # save the fitted model
    def save_model(self):
        from joblib import dump
        dump(self, './models/rfclassifier.joblib')
