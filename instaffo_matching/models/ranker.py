import asyncio
from typing import Dict, Tuple, Optional, Type
import logging
from logging.handlers import RotatingFileHandler
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from instaffo_matching.features.engineer import FeatureEngineer

# Configure logger
logger = logging.getLogger(__name__)

class ModelStrategy(BaseEstimator):
    """Base class for model strategies."""
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

class GradientBoostingStrategy(ModelStrategy):
    """Gradient Boosting model strategy."""
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class FeatureEngineerFactory:
    """Factory for creating feature engineers."""
    @staticmethod
    def create(engineer_type: str) -> FeatureEngineer:
        if engineer_type == "default":
            return FeatureEngineer()
        # For future extensions
        raise ValueError(f"Unknown engineer type: {engineer_type}")

class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass

class TalentJobRanker:
    """
    A class used to train and use a machine learning model for ranking the match between talents and jobs.

    This class implements the Singleton pattern to ensure only one instance exists.

    Attributes:
        model (ModelStrategy): The machine learning model strategy used for predictions.
        feature_engineer (FeatureEngineer): The object used to transform raw data into model-ready features.
    """
    _instance = None

    def __new__(cls, model_path: Optional[str] = None, model_strategy: Type[ModelStrategy] = GradientBoostingStrategy):
        if cls._instance is None:
            cls._instance = super(TalentJobRanker, cls).__new__(cls)
            cls._instance._initialize(model_path, model_strategy)
        return cls._instance

    def _initialize(self, model_path: Optional[str], model_strategy: Type[ModelStrategy]):
        """
        Initializes the TalentJobRanker with an optional pre-trained model.

        Args:
            model_path (str, optional): The path to a pre-trained model. If not provided, initializes a new model.
            model_strategy (Type[ModelStrategy]): The model strategy to use.
        """
        self.model_strategy = model_strategy()
        self.feature_engineer = FeatureEngineerFactory.create("default")
        if model_path:
            asyncio.run(self._load_model(model_path))
        else:
            logger.info("Initialized a new %s model and FeatureEngineer.", model_strategy.__name__)

    async def _load_model(self, model_path: str):
        """
        Asynchronously loads the model and feature engineer from the specified path.

        Args:
            model_path (str): The path to the pre-trained model.

        Raises:
            ModelLoadError: If loading the model or feature engineer fails.
        """
        try:
            self.model_strategy = await asyncio.to_thread(joblib.load, model_path)
            self.feature_engineer = await asyncio.to_thread(joblib.load, model_path.replace("model_", "feature_engineer"))
            logger.info("Loaded model and feature engineer from %s", model_path)
        except FileNotFoundError as e:
            logger.error("File not found: %s", e)
            raise ModelLoadError(f"File not found: {e}")
        except Exception as e:
            logger.error("Failed to load model or feature engineer: %s", e)
            raise ModelLoadError(f"Failed to load model: {e}")

    def fit(self, talent_df: pd.DataFrame, job_df: pd.DataFrame, labels: pd.DataFrame):
        """
        Fits the model using the provided talent and job data.

        Args:
            talent_df (pd.DataFrame): DataFrame containing talent data.
            job_df (pd.DataFrame): DataFrame containing job data.
            labels (pd.DataFrame): DataFrame containing match labels.

        Returns:
            None
        """
        logger.info("Starting training process.")
        try:
            X_train, X_test, y_train, y_test = self._prepare_data(talent_df, job_df, labels)
            self._train_model(X_train, y_train)
            self._evaluate_model(X_test, y_test)
        except Exception as e:
            logger.error("Error during the training process: %s", e)
            raise

    def _prepare_data(self, talent_df: pd.DataFrame, job_df: pd.DataFrame, labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the training and testing data.

        Args:
            talent_df (pd.DataFrame): DataFrame containing talent data.
            job_df (pd.DataFrame): DataFrame containing job data.
            labels (pd.DataFrame): DataFrame containing match labels.

        Returns:
            Tuple containing the training features, testing features, training labels, and testing labels.
        """
        job_df_train, job_df_test, talent_df_train, talent_df_test = train_test_split(
            job_df, talent_df, test_size=0.2, random_state=42, stratify=labels["label"]
        )
        # Fit on training data and transform both training and testing data
        X_train = self.feature_engineer.fit_transform(job_df_train, talent_df_train)
        X_test = self.feature_engineer.transform(job_df_test, talent_df_test)
        
        # Get labels for training and testing data
        y_train = labels.loc[job_df_train.index, "label"]
        y_test = labels.loc[job_df_test.index, "label"]
        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the model with the provided data.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        self.model_strategy.fit(X_train, y_train)
        logger.info("Model training completed.")

    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluates the model and logs the results.

        Args:
            X_test (np.ndarray): Testing feature matrix.
            y_test (np.ndarray): True labels for the testing set.
        """
        y_pred = self.model_strategy.predict(X_test)
        logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    def predict(self, job: pd.DataFrame, talent: pd.DataFrame) -> Tuple[bool, float]:
        """
        Synchronously predicts the match label and score for given talent and job profiles.

        Args:
            talent (pd.DataFrame): DataFrame containing the talent's profile.
            job (pd.DataFrame): DataFrame containing the job's profile.

        Returns:
            Tuple[bool, float]: The predicted label (True/False) and score (float).

        Example:
            >>> ranker = TalentJobRanker()
            >>> talent = pd.DataFrame({"languages": ..., "job_roles": ...})
            >>> job =  pd.DataFrame({"languages": ..., "job_roles": ...})
            >>> label, score = ranker.predict(job, talent)
            >>> print(f"Match: {label}, Score: {score}")
            Match: True, Score: 0.85
        """
        try:
            # Directly call the synchronous methods
            features = self._transform_input(job, talent)
            labels = self.model_strategy.predict(features)
            scores = self.model_strategy.predict_proba(features)
            logger.info("Prediction made successfully.")
            # Check if the input is for a single prediction or multiple predictions
            if len(labels) == 1:
                return bool(labels[0]), float(scores[0][1])
            else:
                return [bool(label) for label in labels], [float(score[1]) for score in scores]
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            raise

    def _transform_input(self, talent: pd.DataFrame, job: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input talent and job profiles into the feature format required by the model.

        Args:
            talent (Dict): Dictionary containing the talent's profile.
            job (Dict): Dictionary containing the job's profile.

        Returns:
            np.ndarray: The transformed feature matrix.
        """
        return self.feature_engineer.transform(talent, job)

    async def save_model(self, model_path: str):
        """
        Asynchronously saves the trained model and feature engineer to a specified path.

        Args:
            model_path (str): The path where the model will be saved.

        Returns:
            None
        """
        try:
            await asyncio.to_thread(joblib.dump, self.model_strategy, model_path)
            await asyncio.to_thread(joblib.dump, self.feature_engineer, model_path.replace("model_", "feature_engineer"))
            logger.info("Model and feature engineer saved to %s", model_path)
        except Exception as e:
            logger.error("Error saving the model or feature engineer: %s", e)
            raise