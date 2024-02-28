# Loan Prediction Modeling

This project builds classification models to predict loan approval status based on application data.

## Data

The loan application data is in `loan_data.csv`. It contains the following features:

- `Loan_ID`: Unique loan ID
- `Gender`: Male/Female
- `Married`: Yes/No  
- `Dependents`: Number of dependents
- `Education`: Graduate/Not Graduate
- `Self_Employed`: Yes/No
- `ApplicantIncome`: Applicant's income
- `CoapplicantIncome`: Co-applicant's income
- `LoanAmount`: Loan amount in thousands
- `Loan_Amount_Term`: Term of loan in months
- `Credit_History`: credit history meets guidelines
- `Property_Area`: Urban/Semi-Urban/Rural
- `Loan_Status`: Loan approved (Y/N)

## Models

The following models are evaluated:

- Random Forest Classifier
- XGBoost Classifier

Model performance is evaluated on test set accuracy and cross-validation accuracy.

The modeling pipeline includes:

- Data loading and cleaning
- Exploratory data analysis
- Feature encoding and preprocessing 
- Train-test split
- Model training, prediction and evaluation

## Results

On the test set, Random Forest achieves 82.26% accuracy, outperforming XGBoost at 79.03% accuracy.

5-fold cross-validation shows Random Forest has 83.34% average accuracy compared to 78.46% for XGBoost.

## Conclusion
In this project, classification models were built to predict loan approval using Random Forest and XGBoost. Random Forest performed better with 82.26% accuracy. The model pipeline provides a good foundation and can be extended by trying different algorithms and tuning hyperparameters. Additional feature engineering may also help improve performance further.

## Next Steps

- Try different classification algorithms
- Tune model hyperparameters 
- Engineer new features
- Reduce overfitting through regularization

## References

The dataset can be found on [Loan Status Prediction](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction?select=loan_data.csv) and python code for this project can be found [HERE](https://github.com/sihlemsk/DataScienceProjects/blob/main/Classifier%20Projects/Loan%20Status%20Prediction%20(Siphesihle%20Masuku).ipynb)
