# Telecom-Customer-churn

## Project/Goals
The goal of this project is to analyze customer churn in the telecommunications industry. We aim to understand the factors influencing churn and develop churn prediction models. By exploring customer demographics, factors influencing churn, and evaluating different prediction models, we want to provide insights and recommendations to reduce customer churn for telecom companies.

## Process
# Step 1: Data Collection and Preprocessing

About Dataset
A telco company that provided home phone and Internet services to 7043 customers in California

Conducted data preprocessing tasks such as handling missing values, encoding categorical variables, and scaling numerical features.

# Step 2: Exploratory Data Analysis (EDA)
> Performed EDA to gain insights into the data and understand the characteristics of churned and non-churned customers.
> Analyzed customer demographics, service usage patterns, contract types, and payment information to identify potential churn predictors.
Below are some observations:
> Customers who do not subscribe to additional online security services tend to churn (31.3% have left), while only 14.6% of customers with additional online security services have left.
> Almost one-third (31.2%) of customers without a technical support plan have left, compared to just 15.2% of those who have.
> Nearly half (41.7%) of elderly customers have left, compared to 23.6% of non-elderly customers. Senior citizen customers are more likely to churn.
> Customers without a partner have a higher chance of churn (33% of customers without a partner have left).
> Customers without children, parents, or grandparents are prone to churn (32.6% of them have left).
> Customers who have been with the company for a shorter duration are more likely to churn.


# Step 3: Feature Engineering
Selected the relevent features that can be helpful in predicting customer churn. Below features are used 

Senior Citizen , Partner,	Dependents	, Tenure Months	, Phone Service,	Multiple Lines,	Internet Service,	Online Security,	Online Backup,	Device Protection	,Tech Support,	Streaming TV,	Streaming Movies,	Contract	,Paperless Billing,	Payment Method,	Monthly Charges

# Step 4: Model Development
Developed and evaluated different churn prediction models using machine learning algorithms - Random Forest Classifier , Logistic Regression , AdaBoost Classifer , XGBoost Classifier
Split the data into training and testing sets, trained the models on the training data, and evaluated their performance using various metrics such as accuracy, precision, recall, and F1-score.

# Step 5: Model Selection and Evaluation
Compared the performance of different models and selected the best-performing model based on evaluation metrics.
Conducted cross-validation and performed hyperparameter tuning to optimize the chosen model.
XGBoost with k-fold cross validation was found to be providing best results
The chosen XGBoost model demonstrates superior performance, with high mean F1 score of 0.85 and overall accuracy of 0.79

# Step 6: Deployement
> Deployment was done by using Flask. 
> User can select the features to get the prediction and suggestion for customer retention
> The portal will provide the probablity of the customer churn


# Conclusion 
> Customer can be segmented based on churn probability
> Targeted marketing campaigns can be created for each segment
> Tailor offers and promotions to meet the specific needs and preferences of each customer segment.
> Offer incentives such as discounts on upgrades, free trial periods, or exclusive rewards for loyal customers.
>  This will help a telecommnication service provider to increase customer retention





