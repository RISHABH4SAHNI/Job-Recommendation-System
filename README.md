# Job Recommendation System

Machine learning-based job recommendation system that classifies job postings and generates personalized recommendations using NLP and ML.

## Features
- **Job Domain Classification**: Automatically classify job postings into domains
- **Skill Relevancy Scoring**: Compute skill and text similarity between candidates and jobs
- **Recommendation Engine**: Generate personalized job recommendations
- **REST API**: FastAPI-based endpoints with OpenAPI docs

## Project Structure
```
src/
  api/
    main.py                # FastAPI app entrypoint
    models/schemas.py      # Pydantic request/response models
    routes/
      classification.py    # /classification endpoints
      recommendations.py   # /recommendations endpoints
  data_processing/
    csv_processor.py       # CSV loaders/cleaners and utilities
  job_domain_classifier/
    classifier.py          # Training, inference, model IO
    preprocessing.py       # Text cleaning and domain assignment
  matching_engine/
    candidate_job_matcher.py
    recommendation_generator.py
  resume_parser/
    llama_parser.py
  skill_relevancy_scorer/
    fuzzy_matcher.py
    relevancy_calculator.py
    skill_extractor.py
  utils/
    file_utils.py
    ml_utils.py

config/
  settings.py              # Paths, API config, env vars
  logging_config.py        # Logging setup

scripts/
  process_data.py          # Organize/process CSV data
  train_models.py          # Train and save classifier

data/
  raw/                     # Place input CSVs (e.g., job_details.csv)
  processed/               # Generated processed datasets
  models/                  # Saved models

tests/                     # Unit tests
```

## Requirements
- Python 3.8+

## Installation
Install in editable mode using the included packaging metadata:
```bash
pip install -e .
```

This project uses configuration from environment variables (optional). Create a `.env` in the project root to override defaults:
```
API_HOST=127.0.0.1
API_PORT=8000
MLFLOW_TRACKING_URI=.\mlflow
```

## Data Setup
Place the required CSV files into `data/raw/`:
- `job_details.csv` (required for recommendations and training)
- Optional: `all_resumes_data.csv`, `job_details_with_predictions.csv`, etc.

You can also run the helper script to organize/process data:
```bash
python scripts/process_data.py
```

## Running the API
Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive docs at `http://127.0.0.1:8000/docs` or visit `/docs` when running the server.

## API Overview
- Note: The classification endpoint requires a trained model saved under `data/models/`. Run training first if you haven't.
- `GET /` – Root service info
- `GET /recommendations/health` – Health check
- `POST /classification/predict` – Predict job domain
  - Request body:
    ```json
    { "description": "We are looking for a Python developer with ML experience" }
    ```
  - Example response:
    ```json
    { "domain": "Software Development", "confidence": 0.8, "cleaned_text": "python developer ml experience" }
    ```
- `POST /recommendations/generate` – Generate personalized job recommendations
  - Request body:
    ```json
    {
      "candidate": {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "skills": ["python", "machine learning", "sql"],
        "experiences": "Worked on data pipelines",
        "education": "B.Tech Computer Science",
        "domains": ["Data Science", "Software Development"]
      },
      "max_recommendations": 5,
      "filters": {"location": "Remote"}
    }
    ```
  - Example response (array):
    ```json
    [
      {"job_id": "123", "title": "Python Developer", "company": "Tech Corp", "location": "Remote", "relevancy_score": 0.92}
    ]
    ```

## Training the Classifier
Ensure `data/raw/job_details.csv` exists, then run:
```bash
python scripts/train_models.py
```
The best model will be saved under `data/models/`.

## Running Tests
```bash
python -m unittest discover -s tests
```

## Docker
Build the image and run the API:
```bash
docker build -t job-reco:dev .
docker run --rm -p 8000:8000 -e API_HOST=0.0.0.0 -e API_PORT=8000 -v %cd%/data:/app/data job-reco:dev
```

Using docker-compose (recommended for development):
```bash
docker compose up --build
```

What this does:
- Builds from `Dockerfile`
- Serves the API on port 8000 with hot reload
- Mounts `./src`, `./config`, `./scripts`, and `./data` into the container

Environment variables (can be set via a `.env` file at project root):
- `API_HOST` (default `0.0.0.0`)
- `API_PORT` (default `8000`)
- `MLFLOW_TRACKING_URI` (optional)

Optional database:
- Not required by default. A Postgres service is stubbed and commented in `docker-compose.yml`. Uncomment and configure if you add persistence later.

## Development Notes
- Use `pip install -e .` to ensure imports like `src.api` resolve correctly during development.
- Default directories are created automatically (`data/raw`, `data/processed`, `data/models`).
- Logging is configured via `config/logging_config.py` and used across scripts and API.
