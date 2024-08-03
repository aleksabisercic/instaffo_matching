# Instaffo Matching System

![Instaffo Logo](assets/instaffo_logo.png)

[![Coverage Status](https://img.shields.io/codecov/c/github/username/instaffo_matching/master.svg)](https://codecov.io/gh/username/instaffo_matching)
[![PyPI version](https://badge.fury.io/py/instaffo-matching.svg)](https://badge.fury.io/py/instaffo-matching)


Instaffo Matching is Talent-Job Matching System designed to efficiently match job canidats with suitable job opportunities. Utilizing advanced machine learning techniques, natural language processing, and a multi-stage matching pipeline, this system provides highly accurate and explainable matches between talents and jobs.

## Table of Contents

- [Instaffo Matching System](#instaffo-matching-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
      - [**Key Classes:**](#key-classes)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Quick Start](#quick-start)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
    - [Training the Model](#training-the-model)
    - [Evaluating the Model](#evaluating-the-model)
  - [API Usage](#api-usage)
    - [Advanced Usage](#advanced-usage)
    - [Support](#support)

## Project Overview

[Raw Data] -> [Initial Filtering] -> [Feature Engineering] -> [ML Prediction] -> [Ranking] -> [Explanation] -> [Final Matches]

Instaffo Matching employs a sophisticated multi-stage pipeline to ensure optimal matching between talents and jobs:

1. **Initial Filtering**: Quickly eliminates incompatible matches based on essential criteria.
2. **Feature Engineering**: Transforms raw data into meaningful features using TF-IDF embeddings and custom transformations.
3. **Machine Learning Prediction**: Utilizes gradient boosting model to predict match quality.
4. **Ranking**: Sorts potential matches based on predicted scores and additional criteria.
5. **(Comming soon) Explanation**: Explanability into matching decisions using SHAP values.

## Features

- Advanced feature engineering using Tf-Idf embeddings and other
- Multi-stage search pipeline for efficient and accurate matching
- Explainable AI integration using SHAP (SHapley Additive exPlanations)
- Asynchronous bulk matching capabilities
- RESTful API with OpenAPI (Swagger) specification
- Comprehensive logging and monitoring
- Scalable architecture ready for high-volume processing

#### **Key Classes:**
- `FeatureEngineer`: Handles data preprocessing and feature extraction. (`features/engineer.py`)
- `TalentJobRanker`: Manages model training, prediction, and lifecycle. (`models/ranker.py`)
- `CandidateFilter`: Quickly eliminates incompatible matches based on essential criteria. (`models/retriver.py`)
- `Search`: Orchestrates the end-to-end matching process. (`search/search.py`)

## Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aleksabisercic/instaffo_matching.git
   cd instaffo_matching
   ```
2. Install core dependencies:
    ```bash
    pip install -e .
    ```
3. (Optional) Install development for notebooks and visualization tools:
    ```bash
    pip install -e .[dev]
    ```

## Usage

### Quick Start

```python
from instaffo_matching import Search

search = Search()

# Define the talent's profile
talent = {
    "degree": "bachelor",
    "job_roles": ["frontend-developer", "backend-developer"],
    "languages": [
        {"rating": "C2", "title": "German"},
        {"rating": "C2", "title": "English"},
        {"rating": "B2", "title": "French"}
    ],
    "salary_expectation": 48000,
    "seniority": "junior"
}

# Define the job's requirements
job = {
    "job_roles": ["frontend-developer"],
    "languages": [
        {"title": "German", "rating": "C1", "must_have": True},
        {"title": "English", "rating": "B2", "must_have": True}
    ],
    "max_salary": 70000,
    "min_degree": "none",
    "seniorities": ["junior", "midlevel"]
}

# Perform the matching process
result = search.match(talent, job)

# Print the match results
print(f"Match Result: {result['label']}")
print(f"Match Score: {result['score']:.2f}")
```

## Command Line Interface (CLI)

### Training the Model

To train the model using the provided script, run the following command:

```bash
python scripts/train_model.py --data_path data/data.json --model_save_path models_artifacts/
```

### Evaluating the Model

To evaluate the model's performance, use the evaluation script:

```bash
# Usage from root directory:
python scripts/evaluate_model.py --data_path data/data.json --model_path ./models_artifacts/model_03_08_2024.joblib
```

## API Usage

Start the API server:

```bash
uvicorn talent_job_matcher.api.main:app --reload
```

The API documentation will be available at http://localhost:8000/docs.

### Advanced Usage

For more advanced usage, including bulk matching, explanation of results, and customization of the matching process, please refer to our detailed documentation.

### Support

For support, please open an issue on our GitHub issue tracker or contact our support team at aleksabisercic@gmail.com
