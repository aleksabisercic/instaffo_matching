import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Dict

class FeatureEngineer:
    """
    A class to engineer features for the matching/ranking model.

    This class handles:
    - Extraction and transformation of features from talent and job data.
    - Vectorization of textual data using TF-IDF.
    - Scaling and encoding of features for machine learning models.

    Attributes:
        job_role_vectorizer (TfidfVectorizer): Vectorizer for job roles using TF-IDF.
        preprocessor (ColumnTransformer): Preprocessing pipeline for feature scaling and encoding.
        num_feature_names (List[str]): Names of numerical features.
        cat_feature_names (List[str]): Names of categorical features.
        feature_names (List[str]): Names of all features after preprocessing.
    """

    def __init__(self):
        self.job_role_vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.preprocessor = None 
        
        # for feature importance
        self.feature_names = None
        self.num_feature_names = [
            'language_match_score',
            'role_similarity',
            'degree_diff',
            'salary_difference_percentage',
            'max_seniority_diff',
            'salary_expectation',
            'salary_ratio'
        ]
        
        self.cat_feature_names = [
            'talent_seniority',
            'job_max_seniority',
            'job_min_seniority',
            'talent_degree',
            'job_min_degree',
            'seniority_match',
            'salary_comparison',
            'degree_match_highest'
        ]
        
    def fit(self, job_df: pd.DataFrame, talent_df: pd.DataFrame):
        """
        Fits the feature engineer on the provided data.

        - Fits the TF-IDF vectorizer on the combined job and talent roles.
        - Prepares a preprocessing pipeline for scaling and encoding.

        Args:
            job_df (pd.DataFrame): DataFrame containing job data.
            talent_df (pd.DataFrame): DataFrame containing talent data.
        """
        # Fit TF-IDF vectorizer on job roles
        all_job_roles = job_df['job_roles'].apply(' '.join) + ' ' + talent_df['job_roles'].apply(' '.join)
        self.job_role_vectorizer.fit(all_job_roles)
        
        # Engineer features to get all categorical and numerical features
        features, categorical_features = self.engineer_features(job_df, talent_df)
        
        # Define the preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), list(range(features.shape[1]))),
                ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), 
                 list(range(features.shape[1], features.shape[1] + categorical_features.shape[1])))
            ]
        )
        
        # Fit the preprocessor on the combined features
        combined_features = np.hstack((features, categorical_features))
        self.preprocessor.fit(combined_features)
        
        # Get feature names
        cat_encoder = self.preprocessor.named_transformers_['cat']
        cat_feature_names = []
        for i, feature in enumerate(self.cat_feature_names):
            feature_categories = cat_encoder.categories_[i]
            for category in feature_categories:
                cat_feature_names.append(f"{feature}_{category}")
        
        self.feature_names = self.num_feature_names + cat_feature_names
    
    def get_feature_names(self):
        if self.feature_names is None:
            raise ValueError("FeatureEngineer has not been fitted yet.")
        return self.feature_names
    
    def transform(self, job_df: pd.DataFrame, talent_df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the data using the fitted preprocessor and vectorizer.

        Args:
            job_df (pd.DataFrame): DataFrame containing job data.
            talent_df (pd.DataFrame): DataFrame containing talent data.

        Returns:
            np.ndarray: Transformed feature matrix.
        """
        features, categorical_features = self.engineer_features(job_df, talent_df)
        combined_features = np.hstack((features, categorical_features))
        return self.preprocessor.transform(combined_features)
    
    def fit_transform(self, job_df: pd.DataFrame, talent_df: pd.DataFrame) -> np.ndarray:
        """
        Fits the preprocessor and transforms the data.

        Args:
            job_df (pd.DataFrame): DataFrame containing job data.
            talent_df (pd.DataFrame): DataFrame containing talent data.

        Returns:
            np.ndarray: Transformed feature matrix.
        """
        self.fit(job_df, talent_df)
        return self.transform(job_df, talent_df)
        
    def engineer_features(self, job_df: pd.DataFrame, talent_df: pd.DataFrame):
        """
        Engineers features from job and talent data.

        - Computes numerical and categorical features.
        - Extracts job role similarity using TF-IDF.

        Args:
            job_df (pd.DataFrame): DataFrame containing job data.
            talent_df (pd.DataFrame): DataFrame containing talent data.

        Returns:
            pd.DataFrame: DataFrame containing engineered features.
        """
        numerical_features = []
        categorical_features = []
        
        for idx in job_df.index:
            job = job_df.loc[idx]
            talent = talent_df.loc[idx]
            
            # Language match score
            language_match_score = self._calculate_language_match(talent['languages'], job['languages'])
            
            # Job role similarity using TF-IDF and cosine similarity
            talent_roles = ' '.join(talent['job_roles'])
            job_roles = ' '.join(job['job_roles'])
            role_similarity = self._calculate_role_similarity(talent_roles, job_roles)
            
            # Salary expectation features
            salary_ratio = talent['salary_expectation'] / job['max_salary']
            salary_comparison = 1 if talent['salary_expectation'] > job['max_salary'] else 0
            salary_difference_percentage = (talent['salary_expectation'] - job['max_salary']) / job['max_salary']
            # salary_competitive = (talent['salary_expectation'] <= job['max_salary']) * calculate_competitiveness_index(job['max_salary'], job['job_roles'])

            # Seniority features
            max_seniority = max(sen for sen in job['seniorities'])
            min_seniority = min(sen for sen in job['seniorities'])
            seniority_match = 1 if talent['seniority'] in job['seniorities'] else 0
            # seniority_match_highest = 1 if talent['seniority'] >= max_seniority else 0
            max_seniority_diff = max_seniority - talent['seniority']
            
            # Degree features
            degree_match_highest = 1 if talent['degree'] > job['min_degree'] else 0
            degree_diff = talent['degree'] - job['min_degree']
            

            # Collect numerical features
            numerical_feature_vector = [
                language_match_score,
                role_similarity,
                degree_diff,
                salary_difference_percentage,
                max_seniority_diff,
                talent['salary_expectation'],
                salary_ratio
            ]
            
            # Collect categorical features
            categorical_feature_vector = [
                talent['seniority'],
                max_seniority,
                min_seniority,
                talent['degree'],
                job['min_degree'],
                seniority_match,
                salary_comparison,
                degree_match_highest
            ]
            
            numerical_features.append(numerical_feature_vector)
            categorical_features.append(categorical_feature_vector)
                
        return np.array(numerical_features), np.array(categorical_features)
    
    def _calculate_language_match(self, talent_languages: List[Dict], job_languages: List[Dict]) -> float:
        """
        Calculates the language match score between talent and job. Since must-have languages are
        already filtered, this function computes the match score based on the ratings of prefered and
        required languages.

        Args:
            talent_languages (List[Dict]): List of languages with ratings for the talent.
            job_languages (List[Dict]): List of languages with ratings for the job.

        Returns:
            float: Normalized language match score.
        """
        talent_lang_dict = {lang['title']: lang['rating'] for lang in talent_languages}
        job_lang_dict = {lang['title']: lang['rating'] for lang in job_languages}

        match_score = 0
        for lang, required_level in job_lang_dict.items():
            if lang in talent_lang_dict:
                talent_level = talent_lang_dict[lang]
                if talent_level >= required_level:
                    match_score += 1
                else:
                    match_score += 0.5  # Partial match 

        return match_score / len(job_lang_dict) if job_lang_dict else 0
    
    def _calculate_role_similarity(self, talent_roles: str, job_roles: str) -> float:
        """
        Calculates the similarity between talent and job roles using TF-IDF and cosine similarity.

        Args:
            talent_roles (str): Concatenated string of talent's job roles.
            job_roles (str): Concatenated string of job's roles.

        Returns:
            float: Cosine similarity between the talent and job roles.
        """
        talent_vector = self.job_role_vectorizer.transform([talent_roles])
        job_vector = self.job_role_vectorizer.transform([job_roles])
        return cosine_similarity(talent_vector, job_vector)[0][0]