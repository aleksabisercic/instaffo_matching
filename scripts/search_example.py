import argparse
import asyncio
import json

from instaffo_matching.data.loader import get_matching_dataframes, load_data
from instaffo_matching.search.search import Search
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

    search = Search(model_path)

    # test the match function
    logger.info("Running match function")
    result = search.match(talent=talent_df.iloc[0].to_dict(), job=job_df.iloc[0].to_dict())
    logger.info("Match function result:")
    logger.info(json.dumps(result))

    import time

    t1 = time.time()
    # test the match_bulk function
    results = await search.match_bulk(
        talent_df.head(100).to_dict(orient="records"), job_df.head(100).to_dict(orient="records")
    )
    t2 = time.time()
    logger.info(f"Time taken to run match_bulk: {t2 - t1}. Total results: {len(results)}")
    logger.info("Match bulk function result (first 5):")
    logger.info(json.dumps(results[:10]))
    logger.info("Match bulk function result (last 5):")
    logger.info(json.dumps(results[-10:]))

    #


if __name__ == "__main__":
    # Useage from root:
    # python scripts/search_example.py --data_path data/data.json --model_path ./models_artifacts/model_03_08_2024.joblib
    args = parse_args()
    asyncio.run(main(args.data_path, args.model_path))
