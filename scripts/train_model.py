import argparse
import asyncio
import datetime

from instaffo_matching.data.loader import get_matching_dataframes, load_data
from instaffo_matching.data.preprocessor import standardize_data
from instaffo_matching.models.ranker import TalentJobRanker
from instaffo_matching.utils.logging import setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Train TalentJobRanker model")
    parser.add_argument(
        "--data_path", type=str, default="./data/data.json", help="Path to the data JSON file"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./models_artifacts/",
        help="Directory to save the trained model",
    )
    return parser.parse_args()


async def main(data_path, model_save_path):
    # Load data
    data = load_data(data_path)
    talent_df, job_df, labels_df = get_matching_dataframes(data=data)
    logger.info("Data loaded successfully")

    # Clean / Preprocess data
    talent_df, job_df = standardize_data(talent_df, job_df)
    logger.info("Data cleaned successfully")

    # Fit model
    ranker = TalentJobRanker()
    ranker.fit(talent_df, job_df, labels_df)
    logger.info("Model fitted successfully")

    # Save the trained model
    formatted_time = datetime.datetime.now().strftime("%d_%m_%Y")
    model_path = f"{model_save_path}/model_{formatted_time}.joblib"
    await ranker.save_model(model_path)
    logger.info(f"Model saved successfully at {model_path}")

    # Load models and predict
    sample_talent = talent_df.sample(2)
    sample_job = job_df.loc[sample_talent.index]

    ranker = TalentJobRanker(model_path=model_path)
    logger.info("Loaded model and feature engineer successfully")
    label, score = ranker.predict(sample_job, talent_df)
    logger.info(f"Predicted label: {label}, score: {score}")


if __name__ == "__main__":
    # Useage from root:
    # python scripts/train_model.py --data_path data/data.json --model_save_path models_artifacts/
    args = parse_args()
    asyncio.run(main(args.data_path, args.model_save_path))
