WorkforceIQ – Employee Attrition & HR Analytics Platform

An AI-powered HR Analytics platform that predicts employee attrition, performance ratings, and promotion likelihood using Machine Learning and an interactive Streamlit dashboard.

This project helps HR teams identify employees at risk of leaving, understand workforce patterns, and make data-driven retention decisions.

Project Overview

Employee turnover is a major challenge for organizations because it increases recruitment cost, reduces productivity, and disrupts teams. This project analyzes employee data and builds machine learning models to predict attrition and workforce performance so HR teams can take proactive action. 

Employee_Attrition

The system includes:

Attrition risk prediction

Performance rating forecasting

Promotion likelihood prediction

Interactive HR analytics dashboard

Features
1️⃣ Attrition Risk Prediction

Predicts whether an employee is likely to leave the company.

Model considers factors such as:

Overtime

Salary

Job involvement

Distance from home

Years at company

Job level

Manager relationship

2️⃣ Performance Rating Forecast

Predicts the employee’s future performance rating based on:

Job involvement

Salary hike

Work environment satisfaction

Experience

Relationship with manager

3️⃣ Promotion Likelihood Prediction

Estimates if an employee is ready for promotion based on:

Total working years

Performance rating

Job level

Years in current role

4️⃣ Interactive HR Dashboard

Built with Streamlit, the dashboard includes:

Workforce overview

Attrition statistics

Department distribution

Performance rating analysis

ML prediction tools

Technologies Used

Programming

Python

Libraries

Pandas

NumPy

Scikit-learn

Matplotlib

Imbalanced-learn (SMOTE)

Machine Learning Models

Random Forest Classifier

Gradient Boosting Regressor

Visualization

Matplotlib

Web Application

Streamlit

Machine Learning Workflow

Data Cleaning

Exploratory Data Analysis (EDA)

Feature Engineering

Handling class imbalance using SMOTE

Model training using Random Forest

Model evaluation using classification metrics

Deployment using Streamlit dashboard

Model Evaluation Metrics

The models were evaluated using:

Accuracy

Precision

Recall

F1 Score

AUC-ROC

Confusion Matrix

These metrics help determine how well the model identifies employees who are likely to leave. 

Employee_Attrition

Dataset

Dataset: IBM Employee Attrition Dataset

The dataset includes employee features such as:

Age

Department

Job Role

Monthly Income

Distance From Home

Job Satisfaction

Performance Rating

Years at Company

Work Life Balance

These features help identify patterns related to employee retention and performance. 

Employee_Attrition

Project Structure
Employee-Attrition-ML/
│
├── workforce_dashboard.py      # Streamlit application
├── Employee-Attrition.csv      # Dataset
├── employe_attrition_performance.ipynb  # Data analysis notebook
├── requirements.txt
├── README.md
└── Employee_Attrition.pdf      # Project documentation
Installation

pip install -r requirements.txt
Run the Streamlit App
streamlit run workforce_dashboard.py

The app will open in your browser:

http://localhost:8501
Example Use Case

An HR manager enters employee details such as:

Job role

Salary

Experience

Work environment satisfaction

The system predicts:

Probability of employee leaving

Expected performance rating

Promotion readiness

This allows HR teams to take proactive retention actions.

Business Impact

Using predictive HR analytics can help organizations:

Reduce employee turnover

Improve workforce planning

Optimize recruitment costs

Increase employee satisfaction

Improve organizational productivity

Future Improvements

Possible enhancements:

SHAP explainability for model predictions

Deep learning models

Real-time HR data integration

Cloud deployment (AWS / Azure)

Authentication for HR users

Author

Manikandan

Aspiring Data Scientist | HR Analytics | Machine Learning | Streamlit Developer

License

This project is open-source and available under the MIT License.

If you want, I can also give you 3 things that will make this repo look 10× more professional on GitHub:

A killer GitHub description

requirements.txt

LinkedIn project description recruiters love

All three together make your profile look less like a student project and more like actual work.
