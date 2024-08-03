# Instaffo Matching System

Instaffo Matching is Talent-Job Matching System designed to efficiently match job canidats with suitable job opportunities. Utilizing advanced machine learning techniques, natural language processing, and a multi-stage matching pipeline, this system provides highly accurate and explainable matches between talents and jobs.

## Table of Contents

- [Instaffo Matching System](#instaffo-matching-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Setup](#setup)
      - [Prerequisites](#prerequisites)
    - [Core Installation](#core-installation)
    - [Development and Visualization Tools](#development-and-visualization-tools)
  - [Quick Start](#quick-start)
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

If you plan to use notebooks and visualization features, install the additional development dependencies:

```bash
pip install -e .[dev]
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
