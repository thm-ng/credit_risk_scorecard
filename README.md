# Credit Risk Scorecard — Case Study

End-to-end credit risk analytics: data extraction (SQL), EDA, WoE/IV feature engineering, logistic regression scorecard, credit strategy & interactive Dash dashboard.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run the notebook to build the model and save artifacts
jupyter notebook credit_risk_case_study.ipynb
# Execute all cells — this downloads data, trains the scorecard, and saves to data/model_artifacts.pkl

# 2. Launch the dashboard
python dashboard/app.py
# Open http://localhost:8050
```

## Project Structure

```
├── credit_risk_case_study.ipynb   # Full analysis notebook (SQL, EDA, scorecard, strategy)
├── dashboard/
│   └── app.py                     # Plotly Dash dashboard (3 tabs, interactive)
├── data/                          # Auto-generated: SQLite DB + model artifacts
├── Dockerfile                     # Container for AWS deployment
├── requirements.txt
└── README.md
```

## Dashboard Tabs

| Tab | Content |
|---|---|
| Model Performance | ROC curve, score distribution, Information Value chart |
| Portfolio Analysis | Bad rate by age, utilization, delinquency history |
| Credit Strategy | Interactive cut-off slider, approval vs bad rate, risk-tier pricing |

## Deploy to AWS

```bash
# Option A: EC2 / ECS with Docker
docker build -t credit-risk-dashboard .
docker run -p 8050:8050 credit-risk-dashboard

# Option B: AWS App Runner
# Push image to ECR, create App Runner service pointing to it — auto-scales, no infra management.
```

## Tech Stack

Python, Pandas, Scikit-learn, OptBinning, Plotly, Dash, SQLAlchemy, SQLite, Docker
