import asyncio
import logging
from typing import Dict, List

import pandas as pd

from instaffo_matching.data.loader import get_matching_dataframes, load_data
from instaffo_matching.data.preprocessor import standardize_data
from instaffo_matching.models.ranker import TalentJobRanker
from instaffo_matching.models.retriver import CandidateFilter
from instaffo_matching.utils.metrics import timing_decorator

logger = logging.getLogger(__name__)


class Search:
    """
    A class used to perform search and ranking of talents for job opportunities using a machine learning model.

    Attributes:
        ranker (TalentJobRanker): The machine learning model used for ranking.
        filter (CandidateFilter): The filter used to pre-filter candidates based on basic criteria.
    """

    def __init__(self, model_path: str = "../models_artifacts/model_03_08_2024.joblib"):
        """
        Initializes the Search class with a pre-trained model.

        Args:
            model_path (str): The path to the pre-trained model.
        """
        self.ranker = TalentJobRanker(model_path=model_path)
        self.filter = CandidateFilter()

    @timing_decorator
    def match(self, talent: Dict, job: Dict) -> Dict:
        """
        Matches a single talent to a job using the machine learning model.

        Args:
            talent (Dict): Dictionary containing the talent's profile.
            job (Dict): Dictionary containing the job's profile.

        Returns:
            Dict: A dictionary containing the talent, job, predicted label, and ranking score.

        Example:
            >>> search = Search('../models_artifacts/model_03_08_2024.joblib')
            >>> talent = {"languages": [...], "job_roles": [...], "seniority": "junior", "salary_expectation": 48000, "degree": "bachelor"}
            >>> job = {"languages": [...], "job_roles": ["frontend-developer"], "seniorities": ["junior", "midlevel"], "max_salary": 70000, "min_degree": "none"}
            >>> result = search.match(talent, job)
            >>> print(result)
            {
                "talent": {...},
                "job": {...},
                "label": True,
                "score": 0.85,
            }
        """
        if (
            not self.filter.meets_language_requirements(talent["languages"], job["languages"])
            or not self.filter.degree_sufficient(talent["degree"], job["min_degree"])
            or not self.filter.seniority_sufficient(talent["seniority"], job["seniorities"])
        ):
            return {
                "talent": talent,
                "job": job,
                "label": False,
                "score": 0.0,
            }

        talent_df = pd.DataFrame([talent])
        job_df = pd.DataFrame([job])
        talent_df, job_df = standardize_data(talent_df, job_df)
        label, ranking_score = self.ranker.predict(job=job_df, talent=talent_df)

        return {
            "talent": talent,
            "job": job,
            "label": bool(label),
            "score": float(ranking_score),
        }

    @timing_decorator
    async def match_bulk(self, talents: List[Dict], jobs: List[Dict]) -> List[Dict]:
        """
        Matches multiple talents to multiple jobs using the machine learning model.

        Args:
            talents (List[Dict]): List of dictionaries containing talents' profiles.
            jobs (List[Dict]): List of dictionaries containing jobs' profiles.

        Returns:
            List[Dict]: A list of dictionaries containing the talent, job, predicted label, and ranking score, sorted by score in descending order.

        Example:
            >>> search = Search('../models_artifacts/model_03_08_2024.joblib')
            >>> talents = [{"languages": [...], "job_roles": [...], "seniority": "junior", "salary_expectation": 48000, "degree": "bachelor"}, ...]
            >>> jobs = [{"languages": [...], "job_roles": ["frontend-developer"], "seniorities": ["junior", "midlevel"], "max_salary": 70000, "min_degree": "none"}, ...]
            >>> results = await search.match_bulk(talents, jobs)
            >>> print(results)
            [
                {
                    "talent": {...},
                    "job": {...},
                    "label": True,
                    "score": 0.85,
                },
                ...
            ]
        """
        all_results = []

        for job in jobs:
            filtered_talents = self.filter.filter_candidates(job, talents)
            if not filtered_talents:
                continue

            # Create DataFrames for batch processing
            talent_df = pd.DataFrame(filtered_talents)
            job_df = pd.DataFrame([job] * len(filtered_talents))

            # Standardize data
            talent_df, job_df = standardize_data(talent_df, job_df)

            # Predict in batch
            labels, scores = self.ranker.predict(job=job_df, talent=talent_df)

            # Collect results
            job_results = [
                {
                    "talent": talent,
                    "job": job,
                    "label": label,
                    "score": score,
                }
                for talent, label, score in zip(filtered_talents, labels, scores)
            ]
            all_results.extend(job_results)

        return sorted(all_results, key=lambda x: x["score"], reverse=True)

    def warm_up_cache(self, talents: List[Dict], jobs: List[Dict]):
        """
        Pre-compute and cache results for common queries.

        Args:
            talents (List[Dict]): List of dictionaries containing talents' profiles.
            jobs (List[Dict]): List of dictionaries containing jobs' profiles.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    def update_model(self, new_model_path: str):
        """
        Hot-swap the model without downtime.

        Args:
            new_model_path (str): The path to the new pre-trained model.
        """
        new_ranker = TalentJobRanker(new_model_path)
        self.ranker = new_ranker
        logger.info(f"Model updated to {new_model_path}")
        # self.cache.clear()  # Uncomment if cache is implemented
        logger.info(f"Model updated to {new_model_path}")


# Example usage
if __name__ == "__main__":
    data = load_data("../data/data.json")
    talent_df, job_df, labels_df = get_matching_dataframes(data=data)

    search = Search("../models_artifacts/model_03_08_2024.joblib")

    # Test the match function
    result = search.match(talent=talent_df.iloc[0].to_dict(), job=job_df.iloc[0].to_dict())
    print(result)

    # Test the match_bulk function
    results = asyncio.run(
        search.match_bulk(
            talent_df.head(100).to_dict(orient="records"),
            job_df.head(100).to_dict(orient="records"),
        )
    )
    print(results)
