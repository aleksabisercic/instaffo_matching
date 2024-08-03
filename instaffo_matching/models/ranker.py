from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from instaffo_matching.features.engineer import FeatureEngineer


class TalentJobRanker:
    def __init__(self, model_path: str = None):
        # Load the model and feature engineer if a path is provided
        if model_path:
            self.model = joblib.load(model_path)
            self.feature_engineer = joblib.load(model_path.replace("model", "feature_engineer"))
        else:
            # If no model path provided, set up for new training
            self.model = GradientBoostingClassifier()
            self.feature_engineer = FeatureEngineer()

    def fit(self, talent_df: pd.DataFrame, job_df: pd.DataFrame, labels: pd.DataFrame):
        # split job_df and talent_df into train and test and then fit feature engineer on train and transform on test
        job_df_train, job_df_test, talent_df_train, talent_df_test = train_test_split(
            job_df, talent_df, test_size=0.2, random_state=42, stratify=labels["label"]
        )
        # Fit and transform the feature engineer
        X_train = self.feature_engineer.fit_transform(job_df_train, talent_df_train)
        X_test = self.feature_engineer.transform(job_df_test, talent_df_test)

        y_train = labels.loc[job_df_train.index, "label"]
        y_test = labels.loc[job_df_test.index, "label"]

        # Fit the model
        self.model = GradientBoostingClassifier()
        self.model.fit(X_train, y_train)

        self.evaluate(X_test, y_test)

    def predict(self, talent: Dict, job: Dict) -> tuple:
        # Create dataframes from the input dictionaries
        talent_df = pd.DataFrame([talent])
        job_df = pd.DataFrame([job])
        # Transform features using the previously fitted feature engineer
        features = self.feature_engineer.transform(job_df, talent_df)
        # Predict the match label and score
        label = self.model.predict(features)[0]
        score = self.model.predict_proba(features)[0][1]  # Probability of positive class (match)
        return bool(label), float(score)

    def evaluate(self, features: np.ndarray, labels: np.ndarray):
        y_pred = self.model.predict(features)
        print(confusion_matrix(labels, y_pred))
        print(classification_report(labels, y_pred))
        print("\n")

        # Get feature importances
        gb_importances = self.model.feature_importances_
        feature_names = self.feature_engineer.get_feature_names()

        # Create a DataFrame for better visualization
        gb_feature_importances_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": gb_importances}
        )

        # Sort the DataFrame by importance
        gb_feature_importances_df = gb_feature_importances_df.sort_values(
            by="Importance", ascending=False
        )

        return gb_feature_importances_df

    def save_model(self, model_path: str):
        # Save the model and feature engineer to specified path
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_engineer, model_path.replace("model", "feature_engineer"))
