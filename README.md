# Signal Check

A Flask web interface for a TF-IDF / Passive Aggressive news-text classifier. It returns a label learned from the training dataset; it does not verify claims or establish whether a story is true.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open http://127.0.0.1:5000. Training downloads the configured public Kaggle dataset and writes `model_bundle.joblib`. The app loads this bundle at startup and exits with a clear error if it is missing. Re-run training after changing the model or training code.

## Deploy on Render

`render.yaml` configures a Render web service. Push this repository to GitHub, choose **New > Blueprint** in Render, and connect the repository. Render installs the dependencies, trains the model as part of the build, then starts the app under Gunicorn. The first build can take several minutes, and each later build trains again. The resulting URL appears in the Render dashboard.

KaggleHub downloads public datasets without credentials unless a dataset requires consent or authentication. If the build log reports an authentication or consent error, create a Kaggle API token and add it in the Render service's environment as `KAGGLE_API_TOKEN`. Never commit the token to this repository.

For manual Render setup, use:

- Build command: `pip install -r requirements.txt && python train_model.py`
- Start command: `gunicorn --workers 1 --threads 4 --timeout 30 --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - app:app`

The `/healthz` endpoint is intended for the hosting platform's health check. The app limits article input size, adds browser security headers, does not persist article submissions, and disables Flask debug mode. The hosting provider still receives requests to process them; avoid submitting sensitive text.

## Before relying on its predictions

The app deliberately displays numeric and unfamiliar labels as model labels instead of guessing which class means fake or real. Check the dataset's label definitions, evaluate the model on recent and independently sourced articles, and review errors and bias for the intended audience. Accuracy on a random held-out split is not evidence that the classifier can fact-check current news.
