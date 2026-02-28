#Ticket Classifier (Customer Support) — ML + API + Docker

End-to-end ML pipeline for classifying customer support tickets.
Includes training, evaluation, FastAPI inference service, and Docker deployment.

Problem

Customer support systems receive large volumes of text tickets.
This project builds a machine learning classifier that automatically categorizes incoming tickets.

Example:

"I cannot access my account"
→ account_issue
Dataset

Banking77 dataset (customer support queries across 77 intent classes).

Model

TF-IDF vectorization

Linear classifier (e.g. Logistic Regression / Linear SVM)

Trained with scikit-learn

Model Performance

Accuracy: 0.86
Macro F1-score: 0.86

Evaluated on held-out test set.

Project Structure
src/
  api/        # FastAPI app
  data/       # Data loading utilities
  models/     # Training & prediction logic
  utils/      # Config
tests/
Dockerfile
requirements.txt
Setup (Local)
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

Train model:

python src/models/train.py

Run API:

uvicorn src.api.main:app --reload

Swagger UI:

http://127.0.0.1:8000/docs
Example API Call
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"My card was charged twice"}'
Docker

Build image:

docker build -t ticket-classifier .

Run container:

docker run -p 8000:8000 ticket-classifier
Tech Stack

Python
scikit-learn
FastAPI
Docker
Uvicorn

