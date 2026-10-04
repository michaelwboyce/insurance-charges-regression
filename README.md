# Insurance Charges Regression

A machine-learning regression project using Python and scikit-learn to predict healthcare insurance charges from demographic and lifestyle data.

## Project Goal

Build a regression model that predicts the `charges` column from customer characteristics, evaluate model performance with R², and apply the trained model to unseen validation data.

## Results

- **Model:** Linear Regression
- **Cross-validation:** 5-fold
- **Mean R²:** **0.7451**
- **Project threshold:** R² > 0.65
- **Result:** Threshold exceeded

## Features

The model uses:

- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

## Data Preparation

The workflow includes:

- Removing missing values
- Standardizing inconsistent text values
- Cleaning currency symbols from `charges`
- Removing invalid ages
- Correcting negative values in `children`
- One-hot encoding categorical features
- Scaling numerical features with `StandardScaler`
- Aligning validation columns with training columns

## Modeling Workflow

1. Load and clean `insurance.csv`
2. Separate features and target
3. Encode categorical variables
4. Scale numerical variables
5. Evaluate `LinearRegression` using 5-fold cross-validation
6. Train the model on the full cleaned dataset
7. Process `validation_dataset.csv` using the same transformations
8. Predict unseen insurance charges
9. Apply a minimum predicted charge of $1,000

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook

## Files

- `insurance_charges_regression.ipynb` — complete modeling workflow
- `requirements.txt` — project dependencies
- `.gitignore` — Python ignore rules

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `insurance.csv` and `validation_dataset.csv` in the project folder, then run:

```bash
jupyter notebook insurance_charges_regression.ipynb
```

## Skills Demonstrated

This project demonstrates practical experience with data cleaning, feature engineering, regression modeling, cross-validation, preprocessing consistency, and generating predictions for unseen data.
