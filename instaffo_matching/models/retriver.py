from typing import List, Dict
import pandas as pd

class CandidateFilter:
    @staticmethod
    def rating_to_level(rating: str) -> int:
        levels = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        return levels.get(rating.upper(), 0)

    @staticmethod
    def meets_language_requirements(candidate_languages: List[Dict], job_languages: List[Dict]) -> bool:
        candidate_dict = {lang['title'].lower(): CandidateFilter.rating_to_level(lang['rating']) for lang in candidate_languages}
        for job_lang in job_languages:
            if job_lang.get('must_have', False):
                job_title = job_lang['title'].lower()
                job_rating = CandidateFilter.rating_to_level(job_lang['rating'])
                if candidate_dict.get(job_title, 0) < job_rating:
                    return False
        return True

    @staticmethod
    def filter_candidates(job: Dict, talents: List[Dict]) -> List[Dict]:
        filtered_talents = []
        for talent in talents:
            if (CandidateFilter.meets_language_requirements(talent['languages'], job['languages']) and
                CandidateFilter.degree_sufficient(talent['degree'], job['min_degree']) and
                CandidateFilter.seniority_sufficient(talent['seniority'], job['seniorities'])):
                filtered_talents.append(talent)
        return filtered_talents

    @staticmethod
    def degree_sufficient(talent_degree: str, job_min_degree: str) -> bool:
        degree_hierarchy = {"none": 0, "apprenticeship": 1, "bachelor": 2, "master": 3, "doctorate": 4}
        return degree_hierarchy.get(talent_degree.lower(), 0) >= degree_hierarchy.get(job_min_degree.lower(), 0)

    @staticmethod
    def seniority_sufficient(talent_seniority: str, job_seniorities: List[str]) -> bool:
        seniority_hierarchy = {"none": 0, "junior": 1, "midlevel": 2, "senior": 3}
        return seniority_hierarchy.get(talent_seniority.lower(), 0) >= min(seniority_hierarchy.get(level.lower(), 0) for level in job_seniorities)
    
    
# import pandas as pd
# from typing import List, Dict

# def filter_candidates(job_details: pd.Series, talents_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Filters the talent dataframe based on the given job details.

#     Parameters:
#     job_details (pd.Series): Series containing job criteria.
#     talents_df (pd.DataFrame): DataFrame containing talent profiles.

#     Returns:
#     pd.DataFrame: A filtered DataFrame of talents that match the job criteria.
#     """
#     def rating_to_level(rating: str) -> int:
#         levels = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
#         return levels.get(rating, 0)

#     def meets_language_requirements(candidate_languages: List[Dict], job_languages: List[Dict]) -> bool:
#         """
#         Checks if the candidate meets the language requirements of the job.
#         """
#         candidate_dict = {lang['title']: rating_to_level(lang['rating']) for lang in candidate_languages}
#         for job_lang in job_languages:
#             if job_lang.get('must_have', False):
#                 job_title = job_lang['title']
#                 job_rating = rating_to_level(job_lang['rating'])
#                 if candidate_dict.get(job_title, 0) < job_rating:
#                     return False
#         return True

#     # Filter candidates based on must-have languages
#     talents_df['meets_language_requirements'] = talents_df['languages'].apply(
#         lambda langs: meets_language_requirements(langs, job_details['languages'])
#     )

#     # Further filter based on minimum degree and seniority
#     talents_df['degree_sufficient'] = talents_df['degree'] >= job_details['min_degree']
#     talents_df['seniority_sufficient'] = talents_df['seniority'].apply(
#         lambda x: any(x >= level for level in job_details['seniorities'])
#     )

#     # Combine all conditions to filter the DataFrame
#     filtered_df = talents_df[
#         (talents_df['meets_language_requirements']) &
#         (talents_df['degree_sufficient']) &
#         (talents_df['seniority_sufficient'])
#     ]

#     return filtered_df.drop(columns=['meets_language_requirements', 'degree_sufficient', 'seniority_sufficient'])


# # Load the data and standardize it
# data = load_data("../data/data.json")
# talent_df, job_df, labels_df = get_matching_dataframes(data=data)
# talent_df, job_df = standardize_data(talent_df, job_df)

# # Get 5 random job details
# job_details_series = job_df.sample(5).apply(pd.Series)

# # Apply the filter_candidates function
# filtered_candidates = []
# for _, job in job_details_series.iterrows():
#     filtered_candidates.append(filter_candidates(job, talent_df))


# filtered_candidates[0].head()