## WorkforceIQ – Employee Attrition & HR Analytics Platform

🚀 WorkforceIQ is an AI-powered HR Analytics platform that predicts employee attrition, performance ratings, and promotion likelihood using Machine Learning and an interactive Streamlit dashboard.

This project helps HR teams identify employees at risk of leaving, understand workforce trends, and take data-driven retention decisions.

## 📌 Project Overview

Employee turnover is a major challenge for organizations because it increases recruitment costs, reduces productivity, and disrupts teams.

This project analyzes employee data and builds machine learning models to:

✅ Predict employee attrition
✅ Forecast employee performance ratings
✅ Identify promotion-ready employees
✅ Provide insights through an interactive HR dashboard

According to the project documentation, the goal is to identify key drivers of attrition and support proactive workforce management decisions. 

## Employee_Attrition

⚡ Features
🚨 Attrition Risk Prediction

## Predicts whether an employee is likely to leave the company.

The model analyzes factors like:

⏱ Overtime

💰 Monthly income

🧠 Job involvement

🏠 Distance from home

📆 Years at company

👔 Job level

🤝 Relationship with manager

⭐ Performance Rating Forecast

## Predicts an employee’s future performance rating using:

📊 Job involvement

💵 Salary hike

🌿 Work environment satisfaction

👨‍💼 Experience

🤝 Relationship satisfaction

🚀 Promotion Likelihood Prediction

## Estimates whether an employee is ready for promotion based on:

📈 Total working years

🏆 Performance rating

🧑‍💼 Job level

⏳ Years in current role

📊 Interactive HR Dashboard

## Built using Streamlit, the platform provides:

👥 Workforce overview

📉 Attrition statistics

🏢 Department distribution

⭐ Performance analysis

🤖 Machine learning predictions

🧠 Machine Learning Workflow

1️⃣ Data Cleaning
2️⃣ Exploratory Data Analysis (EDA)
3️⃣ Feature Engineering
4️⃣ Handling class imbalance using SMOTE
5️⃣ Model training using Random Forest
6️⃣ Model evaluation using classification metrics
7️⃣ Deployment using Streamlit

📊 Model Evaluation Metrics

The models were evaluated using:

📌 Accuracy
📌 Precision
📌 Recall
📌 F1 Score
📌 AUC-ROC
📌 Confusion Matrix

These metrics help measure how accurately the model predicts whether employees will stay or leave the organization. 

Employee_Attrition

## 📂 Dataset

Dataset used:Employee Attrition Dataset

Key features include:

👤 Age

🏢 Department

👔 Job Role

💰 Monthly Income

🏠 Distance From Home

😊 Job Satisfaction

⭐ Performance Rating

⏳ Years At Company

⚖ Work Life Balance

These attributes help identify patterns influencing employee turnover and performance. 

Employee_Attrition

## 🗂 Project Structure
Employee-Attrition-ML
│
├── workforce_dashboard.py
├── employe_attrition_performance.ipynb
├── Employee-Attrition.csv
├── Employee_Attrition.pdf
├── requirements.txt
└── README.md
⚙️ Installation

## Clone the repository

git clone https://github.com/mani572001/WorkforceIQ-HR-Analytics/edit/main

Install dependencies

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit app:

streamlit run workforce_dashboard.py

The application will run at:

http://localhost:8501
💡 Example Use Case

An HR manager enters employee information such as:

Job role

Salary

Experience

Job involvement

Work environment satisfaction

## The system predicts:

📊 Probability of employee leaving
⭐ Expected performance rating
🚀 Promotion readiness

This allows HR teams to intervene early and improve employee retention.

## 🎯 Business Impact

Using predictive HR analytics helps organizations:

✔ Reduce employee turnover
✔ Improve workforce planning
✔ Optimize hiring costs
✔ Increase employee satisfaction
✔ Improve productivity

🔮 Future Improvements

Potential enhancements include:

🔍 Explainable AI (SHAP)

☁ Cloud deployment (AWS / Azure)

🔐 Authentication for HR users

📡 Real-time HR data integration

🤖 Advanced ML / Deep Learning models

👨‍💻 Author

Manikandan

📊 Data Science & Analytics
💻 Machine Learning Projects
🎨 Streamlit Dashboard Development

📜 License

This project is licensed under the MIT License.

One more brutally honest tip before you run off to paste this into GitHub:

A good README is nice, but what actually makes recruiters click your repo is:

Screenshots of the Streamlit dashboard

Demo GIF

Live Streamlit deployment

Those three things turn a repo from “student homework” into “actual product.”
