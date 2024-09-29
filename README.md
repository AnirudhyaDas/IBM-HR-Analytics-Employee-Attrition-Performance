# IBM-HR-Analytics-Employee-Attrition-Performance

# Overview

Employee attrition, or turnover, is a critical concern for organizations, affecting both productivity and profitability. High attrition rates can result in increased costs for recruitment and training, disrupt team dynamics, and lead to the loss of institutional knowledge. This analysis aims to understand the patterns and factors leading to employee attrition by using clustering techniques to segment employees based on demographic, financial, and job-related characteristics. Through this unsupervised machine learning approach, we seek to uncover hidden patterns that contribute to attrition, providing actionable insights for better employee retention strategies.

# Project Objectives
1. Understand Current Turnover Rates: Analyze employee turnover rates by demographic factors like age, gender, job role, and department.
2. Identify Key Factors Influencing Turnover: Use data features such as job satisfaction, salary, and stock options to find correlations with attrition.
3. Build Predictive Models: Train machine learning models to predict employee attrition and optimize them using hyperparameter tuning.

# Table of Contents
- Project Objectives
- Dataset
- Requirements
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Models
- Conclusion
- Contributing

# Dataset
- Source: The dataset contains various features related to employees, such as:
  -Age
  -Gender
  -Department
  -Monthly Income
  -Job Satisfaction
  -Stock Option Level
  -Attrition (Target variable: 0 = No, 1 = Yes)

# Requirements
The following libraries are required for running the notebook:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
# Exploratory Data Analysis (EDA)
Key Findings:
Attrition by Age: Younger employees tend to have higher attrition rates.
Monthly Income vs Attrition: Employees with lower income levels are more likely to leave.
Job Satisfaction vs Attrition: Lower job satisfaction scores correlate with higher turnover rates.

### Code Snippets:
Here are some key visualizations:

1. Attrition by Age
```python
   plt.figure(figsize=(10,6))
   sns.lineplot(data=df, x='Age', hue='Attrition', estimator=None, lw=2)
   plt.title('Attrition by Age')
   plt.xlabel('Age')
   plt.ylabel('Count')
   plt.show()
```
2. Monthly Income vs Attrition
```python
  plt.figure(figsize=(10,6))
  sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='viridis')
  plt.title('Attrition vs Monthly Income')
  plt.show()
```

# Feature Engineering
Before building the models, several steps were taken to prepare the data:

- Handling missing values
- Encoding categorical variables using LabelEncoder
- Scaling numerical features using StandardScaler
- Dropping unnecessary columns and highly correlated features.

# Machine Learning Models
Several models were built and optimized to predict employee attrition. Below is a summary of the models used and their performance after hyperparameter tuning.

### 1. Logistic Regression
```python
  log_reg = LogisticRegression()
  param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga']
  }
```
After hyperparameter tuning, the best model achieved an accuracy of 87.9%.

### 2. Random Forest Classifier
```python
  rf = RandomForestClassifier()
  param_grid_rf = {
      'n_estimators': [100, 200, 300],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
  }
```
The optimized Random Forest model achieved an accuracy of 83.4%.

### 3. Support Vector Classifier (SVC)

```python
  svc = SVC()
  param_grid_svc = {
      'C': [0.1, 1, 10, 100],
      'gamma': [1, 0.1, 0.01, 0.001],
      'kernel': ['linear', 'rbf', 'poly']
  }
```
The best SVC model reached an accuracy of 84%.

### 4. XGBoost Classifier

```python
  xgb = XGBClassifier()
  param_grid_xgb = {
      'n_estimators': [100, 200, 300],
      'max_depth': [3, 5, 7, 10],
      'learning_rate': [0.01, 0.1, 0.2],
      'subsample': [0.8, 1.0]
  }
```
XGBoost performed well with an accuracy of 85.2%.

# Conclusion
The analysis reveals that factors such as age, monthly income, and job satisfaction significantly influence employee attrition. The XGBoost model performed the best in predicting attrition. With these insights, companies can implement targeted retention strategies for employees at risk of leaving.

# Contributing
Contributions are welcome! Please fork this repository and submit a pull request with any improvements or additional features.
