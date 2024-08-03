# Instaffo Matching System

Instaffo Matching is Talent-Job Matching System designed to efficiently match job canidats with suitable job opportunities. Utilizing advanced machine learning techniques, natural language processing, and a multi-stage matching pipeline, this system provides highly accurate and explainable matches between talents and jobs.

## Table of Contents

- [Instaffo Matching System](#instaffo-matching-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
      - [Key Classes](#key-classes)
  - [Setup](#setup)
      - [Prerequisites](#prerequisites)
    - [Core Installation](#core-installation)
    - [Development and Visualization Tools](#development-and-visualization-tools)
  - [Quick Start](#quick-start)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
    - [Training the Model](#training-the-model)
    - [Evaluating the Model](#evaluating-the-model)
  - [API Usage](#api-usage)
    - [Advanced Usage](#advanced-usage)
    - [Support](#support)

## Project Overview

Here I will put Arhitecture

## Features

- Advanced feature engineering using Tf-Idf embeddings and other
- Multi-stage search pipeline for efficient and accurate matching
- Explainable AI integration using SHAP (SHapley Additive exPlanations)
- Asynchronous bulk matching capabilities
- RESTful API with OpenAPI (Swagger) specification
- Comprehensive logging and monitoring
- Scalable architecture ready for high-volume processing

#### Key Classes

The project includes several key classes that are crucial for its functionality:


- **`FeatureEngineer`**: This class is responsible for feature extraction and transformation, significantly contributing to the model's performance. It includes methods for data preprocessing, including handling categorical and numerical features.

  ```python
  from instaffo_matching.features.engineer import FeatureEngineer
  ```
- **`TalentJobRanker`**: This class handles ranking/classification model training, prediction, and other operations. It is the central class for managing model lifecycle and predictions. Async methods in `TalentJobRanker` are used to perform I/O-bound operations, such as loading and saving models, without blocking the main thread. 
  
  ```python
  from instaffo_matching.models.ranker import TalentJobRanker
  ```
- **`Search`** Class: Manages the end-to-end process of filtering and ranking talent-job matches. It includes methods for both single and bulk matching, supporting efficient evaluation and scoring.. It leverages `CandidateFilter` for pre-filtering candidates and `TalentJobRanke`r for predictions.

  ```python
  from instaffo_matching.search.search import Search
  ```

## Setup

#### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

### Core Installation

To install the core dependencies and the package itself, run:
```
git clone https://github.com/aleksabisercic/instaffo_matching.git
cd instaffo_matching
pip install -e .
```

### Development and Visualization Tools

If you plan to use **notebooks** and **visualization features**, install the additional development dependencies:

```bash
pip install -e .[dev]
```

## Quick Start

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
