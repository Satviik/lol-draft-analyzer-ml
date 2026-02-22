# League of Legends Draft Analyzer

A Machine Learning powered drafting assistant designed to analyze champion compositions and recommend optimal picks during League of Legends ranked draft phase.

The system predicts team win probability using historical ranked match data and patch-specific performance statistics.

---

## Features

* Drag-and-drop drafting interface.
* Real-time win probability prediction.
* Champion recommendation engine.
* Patch-aware machine learning model.
* Role-based filtering system.
* Cloud-hosted backend inference.

---

## Live Architecture

Frontend (React + Vercel)

↓

HTTPS Tunnel (Ngrok)

↓

FastAPI Backend (AWS EC2)

↓

XGBoost ML Model

↓

Riot Games Match Dataset

---

## Technology Stack

### Machine Learning

Python
XGBoost
Pandas
NumPy
Scikit Learn

### Backend

FastAPI
Pydantic
Uvicorn

### Infrastructure

AWS EC2 Ubuntu Server
tmux session management
Ngrok HTTPS tunneling

### Frontend

React
TailwindCSS
dnd-kit Drag and Drop

Deployment via Vercel.

---

## Data Pipeline

Riot API Match Collection → Raw JSON Storage → Parsing → Patch Statistics → Role Statistics → Dataset Generation → Model Training → Cloud Deployment.

---

## Machine Learning Approach

The model evaluates:

* Champion Synergy
* Role Matchups
* Patch Meta Strength
* Cross Team Composition Interaction

Objective:

Predict Blue Team Win Probability.

---

## Recommendation Engine

When 9 champions are selected:

1. Remaining role candidates simulated.
2. Model predicts win probability.
3. Delta improvement calculated.

Top champions ranked and returned.

---

## Model Training Details

The machine learning training methodology, feature engineering process, and dataset preparation pipeline are documented separately.

See:

👉  [Model Training Documentation](MODEL_TRAINING.md)

---

## Cloud Backend

Hosted on AWS EC2.

Responsibilities:

* Continuous match ingestion.
* Model inference hosting.
* Patch update retraining pipeline.

---

## Continuous Patch Updates

League patches update every 1–2 weeks.

Pipeline allows:

Collect → Parse → Retrain → Deploy Updated Model.

---

## Architecture Diagram

+--------------------+
| Riot API           |
+--------------------+
          |
          v
+--------------------+
| EC2 Match Collector|
| collect_matches.py |
+--------------------+
          |
          v
+--------------------+
| Raw JSON Dataset   |
+--------------------+
          |
          v
+--------------------+
| Parsing Pipeline   |
+--------------------+
          |
          v
+--------------------+
| Patch + Role Stats |
+--------------------+
          |
          v
+--------------------+
| XGBoost Training   |
+--------------------+
          |
          v
+--------------------+
| FastAPI Backend    |
| AWS EC2            |
+--------------------+
          |
          v
+--------------------+
| Ngrok HTTPS Tunnel |
+--------------------+
          |
          v
+--------------------+
| React Frontend     |
| Vercel Deployment  |
+--------------------+

---

## Future Work

* Automated patch detection.
* Permanent HTTPS domain deployment.
* Scheduled retraining automation.

---

## Author

Satvik Singh Rathore
