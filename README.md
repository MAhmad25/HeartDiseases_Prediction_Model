# Heart Diseases Prediction Project

This project aims to develop a predictive model for heart diseases using various machine learning techniques. The focus is on data cleaning, feature engineering, model training, and deployment of a user-friendly application for real-time risk assessment.

## Data Cleaning

- Handled missing values in `Cholesterol` and `RestingBP` by replacing zeros with mean values
- Verified no duplicate records existed in the dataset
- Confirmed target variable `HeartDisease` was balanced (equal distribution of 0s and 1s)

## Feature Engineering

- Encoded categorical variables (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) using one-hot encoding
- Created `Cholesterol_Level` bins: Desirable (≤200), Borderline High (200-240), High (>240)
- Applied chi-square test to identify statistically significant features (α=0.05)
- Dropped `ChestPainType_TA` feature due to high p-value

## Feature Scaling

- Used StandardScaler to normalize numerical features: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`
- Ensured consistent scaling across training and test datasets

## Model Training & Evaluation

Trained and compared 5 classification models:

- Logistic Regression
- **KNN (K-Nearest Neighbors)** ✓ Best performer
- Decision Tree
- Naive Bayes
- Support Vector Machine

**Results:**
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **KNN** | **87%** | **89%** |

## Deployment

Built interactive Streamlit frontend (`frontend.py`) for real-time predictions:

- User-friendly health metrics input form
- Real-time risk assessment (HIGH RISK / LOW RISK)
- Responsive dark theme UI with custom styling
