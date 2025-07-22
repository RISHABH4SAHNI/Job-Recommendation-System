# Fernglasz Job Recommendation System

A machine learning-based job recommendation system designed to match early talent from Tier 1 colleges with startups seeking interns and full-time hires. This repository contains key components focused on skill relevancy scoring, job domain classification, and matching algorithms developed during my work at Fernglasz.

## Features

* **Skill Relevancy Scoring**

  * Parses resumes using Llama Index (Llama Parse) to extract skills
  * Calculates match scores between student profiles and job descriptions
  * Uses Random Forest models to optimize skill relevancy

* **Job Domain Classification**

  * Categorizes job postings into relevant domains for better recommendations

* **Matching Engine**

  * Matches students with jobs based on multiple relevance signals
  * Generates ranked job recommendations

* **MLflow Integration**

  * Tracks and manages machine learning experiments and models

## Project Structure

* `skill_relevancy_score_calculator/` — Code to compute skill match scores
* `job_domain_classifier/` — Models and scripts for classifying job domains
* `matching_engine/` — Core logic for matching students and jobs
* `mlflow/` — MLflow configuration and tracking files
* `resume_parsing/` — Scripts utilizing Llama Index for resume parsing

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/RISHABH4SAHNI/Fernglasz-Job-Recommendation-System.git
   cd Fernglasz-Job-Recommendation-System
   ```

2. **Set up environment**

   Install required dependencies (example):

   ```bash
   pip install -r requirements.txt
   ```

3. **Run components**

   Follow individual component README files or scripts to run skill scoring, classification, and matching.

4. **Use MLflow**

   Start MLflow UI to track experiments:

   ```bash
   mlflow ui
   ```

## Example Usage

**Calculate Skill Relevancy Score**

```python
from skill_relevancy_score_calculator import calculate_score

score = calculate_score(student_resume, job_description)
print(f"Relevancy Score: {score}")
```

**Classify Job Domain**

```python
from job_domain_classifier import classify_domain

domain = classify_domain(job_description)
print(f"Job Domain: {domain}")
```

## Technologies Used

* Python 3.x
* scikit-learn (Random Forest)
* Llama Index (Llama Parse) for resume parsing
* MLflow for experiment tracking

## Contributing

Contributions and improvements are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

