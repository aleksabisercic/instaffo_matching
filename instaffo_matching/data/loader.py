import json
from typing import List, Dict, Tuple
import pandas as pd

def load_data(file_path: str) -> List[Dict]:
    """
    Load JSON data from a file.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries representing the loaded data.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except json.JSONDecodeError:
        raise ValueError("Failed to decode JSON.")

def get_matching_dataframes(data: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert the list of dictionaries to separate DataFrames for talents, jobs, and labels.

    Parameters:
        data (List[Dict]): Cleaned data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for talents, jobs, and labels.
    """
    records = [create_record(item) for item in data]
    
    talent_df = pd.DataFrame([record["talent"] for record in records])
    job_df = pd.DataFrame([record["job"] for record in records])
    labels_df = pd.DataFrame([{"label": record["label"]} for record in records])
    
    return talent_df, job_df, labels_df

def create_record(item: Dict) -> Dict:
    """
    Extract relevant fields from a single data record to prepare for DataFrame conversion.

    Parameters:
        item (Dict): A single record from the data list.

    Returns:
        Dict: A structured dictionary ready for DataFrame construction.
    """
    return {
        "talent": {
            "degree": item["talent"]["degree"],
            "job_roles": item["talent"]["job_roles"],
            "languages": item["talent"]["languages"],
            "salary_expectation": item["talent"]["salary_expectation"],
            "seniority": item["talent"]["seniority"]
        },
        "job": {
            "job_roles": item["job"]["job_roles"],
            "languages": item["job"]["languages"],
            "max_salary": item["job"]["max_salary"],
            "min_degree": item["job"]["min_degree"],
            "seniorities": item["job"]["seniorities"]
        },
        "label": item["label"]
    }
    

if __name__ == '__main__':
    data = load_data("../data/cleaned_data.json")
    talent_df, job_df, labels_df = get_matching_dataframes(data)
    print(talent_df.head(2))
    print(job_df.head(2))
    print(labels_df.head(2))