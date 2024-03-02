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
### In-depth Analysis of XGBoost Model: 
To gain a deeper understanding of the XGBoost model's performance, a comprehensive evaluation was conducted. The analysis revealed:
#### High Precision: 
The model exhibited a commendable ability to minimize false positives, ensuring a low rate of legitimate job postings being misclassified as fraudulent. This is crucial for maintaining trust and preventing unnecessary job seekers from being discouraged.
#### Balanced Recall: 
While the model achieved high overall accuracy, it exhibited a slight dip in recall specifically for the minority fraudulent class. This indicates that a small number of genuine job postings might be mistakenly classified as fake.
#### Balancing Trade-offs: 
The project acknowledged the inherent trade-off between precision and recall. Adjusting the classification threshold can potentially improve recall for fraud detection, but this comes at the expense of potentially introducing more false positives, misclassifying legitimate postings as fraudulent.

## Next Steps

Some ways to further improve the model:

- Mitigate class imbalance with techniques like oversampling
- Test the model on new data to examine how well it performs
- Fine-tune neural network architectures for text classification
- Incorporate additional features like job poster profiles etc.
- Deploy the model to a production environment for actual usage

## Model Deployment

Once the model development and evaluation stages are complete, the next step is to deploy the model for practical use. Deployment involves making the trained model available for real-time predictions. Common deployment options include:

### API Integration: 
Develop an API to expose the model's functionality, allowing other applications or systems to send input data and receive predictions.

### Cloud Deployment: 
Host the model on a cloud platform (AWS) for scalability and accessibility.

## Model Usage
Once deployed, the model can be utilized for various applications, including:

### Fraud Detection System: 
Integrate the model into a fraud detection system that automatically screens job postings in real-time, flagging potentially fraudulent ones for further investigation.

### User Interface: 
Develop a user interface that allows users to input job details, and the model provides instant feedback on the likelihood of the job posting being fraudulent.

### Batch Processing: 
Implement batch processing for large datasets, enabling the model to analyze and classify multiple job postings simultaneously.

## Conclusion and Future Considerations:

The project successfully demonstrates the efficacy of leveraging machine learning to combat the pervasive issue of fake job postings. The XGBoost model, with its high accuracy and balanced performance, presents a promising solution in the fight against fraudulent recruitment practices. However, it is crucial to acknowledge that the landscape of online scams is constantly evolving, necessitating continuous monitoring and improvement of such models to maintain their effectiveness. Furthermore, exploring alternative imputation techniques and potentially incorporating additional features, such as website analysis or user reviews, could further enhance the model's ability to identify and eliminate fraudulent job postings, ultimately fostering a safer and more reliable job search experience for all.

## References

The dataset can be found on [real or fake job postings predictions](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/download?datasetVersionNumber=1) and python code for this project can be found [HERE](https://github.com/sihlemsk/DataScienceProjects/blob/main/Classifier%20Projects/Job%20Posting%20Prediction%20(Siphesihle%20Masuku).ipynb)
