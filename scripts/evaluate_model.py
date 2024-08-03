import argparse
import asyncio

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
        "--model_path",
        type=str,
        default="./models_artifacts/model_03_08_2024.joblib",
        help="Path to the trained model",
    )
    return parser.parse_args()


async def main(data_path, model_path):
    # Load data
    data = load_data(data_path)
    talent_df, job_df, labels_df = get_matching_dataframes(data=data)
    logger.info("Data loaded successfully")

    # Clean / Preprocess data
    talent_df, job_df = standardize_data(talent_df, job_df)
    logger.info("Data cleaned successfully")

    # get test data
    _, talent_df_test, _, job_df_test, _, labels_test = train_test_split(
        talent_df, job_df, labels_df, test_size=0.2, random_state=42, stratify=labels_df["label"]
    )

    ranker = TalentJobRanker(model_path=model_path)
    logger.info("Loaded model and feature engineer successfully")
    label, score = ranker.predict(job_df_test, talent_df_test)

    # classification report based on predicted labels
    y_true = labels_test["label"].values
    y_pred = label

    logger.info(f"Classification report: {classification_report(y_true, y_pred)}")


if __name__ == "__main__":
    # Useage from root:
    # python scripts/evaluate_model.py --data_path data/data.json --model_path ./models_artifacts/model_03_08_2024.joblib
    args = parse_args()
    asyncio.run(main(args.data_path, args.model_path))
