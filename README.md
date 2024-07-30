# Talent-Job Matching System

## Overview

The Talent-Job Matching System is a sophisticated, AI-powered platform designed to efficiently match job seekers with suitable job opportunities. Utilizing advanced machine learning techniques, natural language processing, and a multi-stage matching pipeline, this system provides highly accurate and explainable matches between talents and jobs.

## Features

- Advanced feature engineering using BERT embeddings
- Multi-stage model pipeline for efficient and accurate matching
- Explainable AI integration using SHAP (SHapley Additive exPlanations)
- Asynchronous bulk matching capabilities
- RESTful API with OpenAPI (Swagger) specification
- Comprehensive logging and monitoring
- Scalable architecture ready for high-volume processing

## Setup

Clone the repository and install:

```
git clone https://github.com/aleksabisercic/instaffo_matching.git
cd instaffo_matching
pip install -e .
```

## Quick Start

```python
from instaffo_matching import Search

search = Search()

talent = {
    "languages": [{"title": "English", "rating": "C2"}, {"title": "German", "rating": "B2"}],
    "job_roles": ["software engineer", "data scientist"],
    "seniority": "mid",
    "salary_expectation": 80000,
    "degree": "master"
}

job = {
    "languages": [{"title": "English", "rating": "C1", "must_have": True}],
    "job_roles": ["software engineer"],
    "seniorities": ["mid", "senior"],
    "max_salary": 90000,
    "min_degree": "bachelor"
}

result = search.match(talent, job)
print(f"Match Result: {result['label']}")
print(f"Match Score: {result['score']:.2f}")
print(f"Ranking Score: {result['ranking_score']:.2f}")
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
