import pandas as pd
from typing import Tuple

def standardize_data(talent_df: pd.DataFrame, job_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize and preprocesses the talent and job data.

    - Encodes categorical variables (degree, seniority, languages) into numerical values.
    - Normalizes the data for machine learning model compatibility.

    Args:
        talent_df (pd.DataFrame): DataFrame containing talent data.
        job_df (pd.DataFrame): DataFrame containing job data.

    Returns:
        tuple: Two DataFrames, talent_df and job_df, with cleaned and encoded data.
    """
    # Hierarchies for encoding categorical features
    degree_hierarchy = {'none': 0, 'apprenticeship': 1, 'bachelor': 2, 'master': 3, 'doctorate': 4}
    seniority_hierarchy = {'none': 0, 'junior': 1, 'midlevel': 2, 'senior': 3}
    language_hierarchy = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

    # Encoding degrees and seniorities
    talent_df['degree'] = talent_df['degree'].map(degree_hierarchy)
    talent_df['seniority'] = talent_df['seniority'].map(seniority_hierarchy)
    job_df['min_degree'] = job_df['min_degree'].map(degree_hierarchy)
    job_df['seniorities'] = job_df['seniorities'].apply(lambda x: [seniority_hierarchy[sen] for sen in x])

    # Encoding languages
    def encode_languages(lang_list):
        return [{**lang, 'rating': language_hierarchy.get(lang['rating'], 0)} for lang in lang_list]

    talent_df['languages'] = talent_df['languages'].apply(encode_languages)
    job_df['languages'] = job_df['languages'].apply(encode_languages)

    return talent_df, job_df