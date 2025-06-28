# Credit Risk Probability Model

An end-to-end implementation of a credit risk scoring system for Bati Bank's buy-now-pay-later service using eCommerce transaction data.

## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability
The Basel II Accord emphasizes comprehensive risk measurement, requiring financial institutions to maintain robust systems for assessing credit risk. This necessitates:
- Interpretable models that can explain risk factors
- Well-documented decision-making processes
- Transparent scoring methodologies
- Regular model validation and monitoring

### Proxy Variable Necessity
Creating a proxy variable is essential because:
- Direct default labels are unavailable
- Transaction patterns serve as indicators of future creditworthiness
- Business risks include:
  - Potential false positives/negatives in risk assessment
  - Need for continuous model validation
  - Importance of maintaining interpretability

### Model Trade-offs
Simple vs Complex Models:
- Simple models (Logistic Regression with WoE):
  - + High interpretability
  - + Regulatory compliance
  - + Easy to explain to stakeholders
  - - Lower predictive power
- Complex models (Gradient Boosting):
  - + Higher predictive accuracy
  - + Better handling of non-linear relationships
  - - Less interpretable
  - - Higher computational requirements

## Project Structure
```
credit-risk-model/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
├── data/                       # (gitignored) data directory
│   ├── raw/                   # Raw input data
│   └── processed/             # Processed data
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 1.0-eda.ipynb         # Exploratory Data Analysis
│   └── 2.0-feature_engineering.ipynb
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Data loading utilities
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_training.py # Model training pipeline
│   │   └── model_evaluation.py
│   └── api/
│       ├── __init__.py
│       ├── main.py           # FastAPI application
│       └── schemas.py        # Pydantic models
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data.py          # Data processing tests
│   └── test_models.py        # Model tests
├── mlruns/                    # MLflow experiment tracking
├── docker/                    # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MLflow tracking:
```bash
mlflow ui
```

4. Run Jupyter notebook for EDA:
```bash
jupyter notebook notebooks/1.0-eda.ipynb
```

## Development Workflow

1. Feature Engineering:
   - Calculate RFM metrics
   - Create aggregate features
   - Handle missing values
   - Normalize/standardize data

2. Model Development:
   - Train multiple models
   - Perform hyperparameter tuning
   - Track experiments with MLflow
   - Evaluate model performance

3. Deployment:
   - Containerize API with Docker
   - Set up CI/CD pipeline
   - Monitor model performance

## API Documentation

The deployed API provides the following endpoints:

- `/predict`: Predict risk probability for new customers
- `/health`: Check service status
- `/metrics`: View model performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.