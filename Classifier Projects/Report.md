# Fake Job Postings Detection

## Overview
This project aims to build a machine learning model to identify fake job postings by analyzing various features such as job titles, locations, company profiles etc. The model will classify job listings as either fraudulent or legitimate.

## Data
The dataset `fake_job_postings.csv` contains the following columns:

- `job_id`: Unique ID for each job posting
- `title`: Job title 
- `location`: Location of job  
- `department`
- `salary_range`
- `company_profile`
- `description`: Job description
- `requirements`: Skills required
- `benefits`
- `telecommuting`: 1 if telecommuting allowed, 0 otherwise
- `has_company_logo`: 1 if company logo present, 0 otherwise 
- `has_questions`: 1 if screening questions present, 0 otherwise
- `employment_type`: Full-time, part-time etc
- `required_experience`
- `required_education`
- `industry`
- `function`
- `fraudulent`: Target variable, 1 if fraudulent job posting, 0 if legitimate

The dataset has 18,000 job postings out of which about 800 are fake.

## Data Exploration
Initial exploration of the data shows the target variable is imbalanced with only ~5% of the data being fraudulent job postings. 

There are missing values present in some columns like `location`, `department` etc. which need to be handled before model building.

## Data Preprocessing
The steps taken:

- Combined `title` and `description` into a new `text` column for NLP modeling.
- Filled missing values in `location`, `employment_type`, `required_experience` and `required_education` with blank strings.
- Split data into train and test sets with a 80:20 ratio.
- Applied preprocessing like CountVectorizer, TfidfVectorizer, OneHotEncoder on the text and categorical features.

## Model Building
The following machine learning models were evaluated:

- Gradient Boosting Classifier
- Random Forest Classifier  
- XGBoost Classifier
- Neural Network (MLP)

XGBoost Classifier gave the best results with **98.5%** accuracy and **0.79** F1 score on the test set after hyperparameter tuning.

## Model Evaluation
The XGBoost model was analyzed further by plotting the confusion matrix and classification report. It had high precision for both classes with a minor dip in recall for the minority fraudulent class. 

Varying the classification threshold improved the recall for the fraudulent class at the expense of more false positives.

Overall, the model is able to effectively identify fake job postings while maintaining high accuracy.

## Next Steps

Some ways to further improve the model:

- Mitigate class imbalance with techniques like oversampling 
- Fine-tune neural network architectures for text classification
- Incorporate additional features like job poster profiles etc.
- Deploy the model to a production environment for actual usage

## References

The dataset and python code for this project can be found in this [GitHub repo](https://github.com/sihlemsk/DataScienceProjects). 
