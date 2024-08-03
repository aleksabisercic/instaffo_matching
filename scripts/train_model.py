import asyncio
import datetime

from instaffo_matching.utils.logging import setup_logger
from instaffo_matching.data.loader import load_data, get_matching_dataframes
from instaffo_matching.data.preprocessor import standardize_data
from instaffo_matching.models.ranker import TalentJobRanker

logger = setup_logger()

async def main():
    try:        
        data = load_data("../data/data.json")
        talent_df, job_df, labels_df = get_matching_dataframes(data=data)
        logger.info("Data loaded successfully")

        # Clean / Preprocess data
        talent_df, job_df = standardize_data(talent_df, job_df)
        logger.info("Data cleaned successfully")

        ranker = TalentJobRanker(model_path)
        await ranker.fit(talent_df, job_df, labels_df)
        logger.info("Model fitted successfully")

        # Optional: Save the trained model
        formatted_time = current_time.strftime("%d_%m_%Y")
        model_path = f"../models_artifacts/model_{formatted_time}.joblib"
        await ranker.save_model(model_path)
        logger.info("Model saved successfully")

        # Example prediction
        sample_talent = talent_df.iloc[0:2]
        sample_job = job_df.iloc[0:2]
        label, score = await ranker.predict(sample_talent, sample_job)
        logger.info(f"Sample prediction - Match: {label}, Score: {score}")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")

if __name__ == "__main__":
    asyncio.run(main())