# Context #

You are world class Data scientist/ML eng.  with world class design pattern software eng knowledge. 
I'd like you to do the task given below to be able to further assess your skills and knowledge. 

The model you chose and it's accuracy is not all that counts. Code quality (including project structure), dependencies and environment management, documentation (docstrings, comments, README file, etc.) are **equally** important!

We wish you good luck (and also a lot of fun) with the task! 🍀

# OBJECTIVE #

Document doce like world-class engineer

## Matching talents and jobs using machine learning

Instaffo is a recruiting platform that makes money by bringing together hiring companies and talents. Companies offer job opportunities and need the right talents to fill their open job positions, talents on the other hand are looking for new job opportunities.

One core component of the Instaffo platform is the search functionality, which e.g. enables talents to only see relevant job opportunities, the most interesting ones ideally being at the top. This involves:

- Filtering out talents/job opportunities that don't _**match**_ (meaning the job opportunity doesn't fit the talents' profile and should therefore not being recommended to them).
- Ranking all results in a meaningful order.

## Task

In this task you'll create a lightweight search & ranking component using Python and machine learning.

First, let's take a look at the provided data (see _**data.json**_), which contains an array of dictionaries, each having the following properties:

- **talent**: profile information of the talent (detailed description below).
- **job**: profile information of the job (detailed description below).
- **label**: `true` if talent and job _**match**_, else `false`.

The **talent** dictionary contains the following properties:

| Field name         |    Type    |                       Description                       |
| :----------------- | :--------: | :-----------------------------------------------------: |
| degree             |    str     |              Highest degree of the talent               |
| job_roles          | list[str]  |          Job roles the talent is interested in          |
| languages          | list[dict] | Languages the talent speaks with their respective level |
| salary_expectation |    int     |  Salary the talent expects a potential new job to pay   |
| seniority          |    str     |                 Seniority of the talent                 |


The **job** dictionary contains the following properties:

| Field name  |    Type    |               Description               |
| :---------- | :--------: | :-------------------------------------: |
| job_roles   | list[str]  |    Applicable job roles for the job     |
| languages   | list[dict] |    Language requirements of the job     |
| max_salary  |    int     |   Maximum salary the job is offering    |
| min_degree  |    str     |  Minimum degree a talent needs to have  |
| seniorities | list[str]  | Seniorities talents are allowed to have |


#### Example from json

```json
[
  {
    "talent": {
      "languages": [
        {
          "rating": "C2",
          "title": "German"
        },
        {
          "rating": "C2",
          "title": "English"
        },
        {
          "rating": "B2",
          "title": "French"
        },
        {
          "rating": "A2",
          "title": "Turkish"
        }
      ],
      "job_roles": [
        "frontend-developer",
        "backend-developer",
        "full-stack-developer",
        "java-developer",
        "mobile-developer"
      ],
      "seniority": "junior",
      "salary_expectation": 48000,
      "degree": "bachelor"
    },
    "job": {
      "languages": [
        {
          "title": "German",
          "rating": "C1",
          "must_have": true
        },
        {
          "title": "English",
          "rating": "B2",
          "must_have": true
        }
      ],
      "job_roles": ["frontend-developer"],
      "seniorities": ["junior", "midlevel"],
      "max_salary": 70000,
      "min_degree": "none"
    },
    "label": true
  },
  {
    "talent": {
      "languages": [
        {
          "rating": "C2",
          "title": "German"
        },
        {
          "rating": "C2",
          "title": "English"
        },
        {
          "rating": "A2",
          "title": "Spanish"
        },
        {
          "rating": "A2",
          "title": "French"
        }
      ],
      "job_roles": [
        "frontend-developer",
        "full-stack-developer",
        "c-c-developer",
        "mobile-developer"
      ],
      "seniority": "junior",
      "salary_expectation": 44000,
      "degree": "master"
    },
    "job": {
      "languages": [
        {
          "title": "German",
          "rating": "C1",
          "must_have": true
        },
        {
          "title": "English",
          "rating": "B2",
          "must_have": true
        }
      ],
      "job_roles": ["frontend-developer"],
      "seniorities": ["junior", "midlevel"],
      "max_salary": 70000,
      "min_degree": "none"
    },
    "label": true
  }
]
```


### Part 1 of code
# src/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Dict

class FeatureEngineer:
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
        features, categorical_features = self.engineer_features(job_df, talent_df)
        combined_features = np.hstack((features, categorical_features))
        return self.preprocessor.transform(combined_features)
    
    def fit_transform(self, job_df: pd.DataFrame, talent_df: pd.DataFrame) -> np.ndarray:
        self.fit(job_df, talent_df)
        return self.transform(job_df, talent_df)
        
    def engineer_features(self, job_df: pd.DataFrame, talent_df: pd.DataFrame):
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
            
            # Seniority features
            max_seniority = max(sen for sen in job['seniorities'])
            min_seniority = min(sen for sen in job['seniorities'])
            seniority_match = 1 if talent['seniority'] in job['seniorities'] else 0
            # seniority_match_highest = 1 if talent['seniority'] >= max_seniority else 0
            max_seniority_diff = max_seniority - talent['seniority']
            
            # Degree features
            degree_match_highest = 1 if talent['degree'] > job['min_degree'] else 0
            degree_diff = talent['degree'] - job['min_degree']
            
            # Salary competitiveness
            # salary_competitive = (talent['salary_expectation'] <= job['max_salary']) * calculate_competitiveness_index(job['max_salary'], job['job_roles'])

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
        """Calculate language match score based on required languages and levels."""
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
        from sklearn.metrics.pairwise import cosine_similarity
        talent_vector = self.job_role_vectorizer.transform([talent_roles])
        job_vector = self.job_role_vectorizer.transform([job_roles])
        return cosine_similarity(talent_vector, job_vector)[0][0]
    
    
    # def _get_average_salary(self, job_roles: List[str]) -> float:
    #     """Get the average salary for the given job roles."""
    #     relevant_salaries = [self.average_salaries[role] for role in job_roles if role in self.average_salaries]
    #     if relevant_salaries:
    #         return np.mean(relevant_salaries)
    #     return 0