# Job Recommendation System

A machine learning-based job recommendation system that matches candidates with relevant job opportunities using advanced NLP and ML techniques.

## Features
- **Job Domain Classification**: Automatically classify job postings into domains
- **Skill Matching**: Match candidate skills with job requirements
- **Recommendation Engine**: Generate personalized job recommendations
- **REST API**: FastAPI-based endpoints for integration

## Quick Start
```bash
pip install -r requirements.txt
python src/api/main.py
```

Visit `/docs` for API documentation when running the server.

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