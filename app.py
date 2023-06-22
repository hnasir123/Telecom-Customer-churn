from flask import Flask, jsonify, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template


app = Flask(__name__, static_folder='static')


# Load the trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# Load the preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

def preprocess_data(df):
    # Convert string categorical variables into numeric variables
    df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
    df['Contract'] = df['Contract'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})

    for col in ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['Multiple Lines'] = df['Multiple Lines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})

    df['Payment Method'] = df['Payment Method'].map({'Credit card (automatic)': 3, 'Bank transfer (automatic)': 2, 'Mailed check': 1, 'Electronic check': 0})

    df['Internet Service'] = df['Internet Service'].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})

    for col in ['Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})

    return df




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    tenure_months = request.form['tenure_months']
    monthly_charges = request.form['monthly_charges']
    total_charges = request.form['total_charges']

    if not tenure_months or not monthly_charges or not total_charges:
        return render_template('index.html', error_message='Please enter values for Tenure Months, Monthly Charges, and Total Charges.')
    
       
    # Get the feature values from the form
    gender = request.form['gender']
    senior_citizen = request.form['senior_citizen']
    partner = request.form['partner']
    dependents = request.form['dependents']
    tenure_months = request.form['tenure_months']
    phone_service = request.form['phone_service']
    multiple_lines = request.form['multiple_lines']
    internet_service = request.form['internet_service']
    online_security = request.form['online_security']
    online_backup = request.form['online_backup']
    device_protection = request.form['device_protection']
    tech_support = request.form['tech_support']
    streaming_tv = request.form['streaming_tv']
    streaming_movies = request.form['streaming_movies']
    contract = request.form['contract']
    paperless_billing = request.form['paperless_billing']
    payment_method = request.form['payment_method']
    monthly_charges = request.form['monthly_charges']
    total_charges = request.form['total_charges']

    # Create a DataFrame with the input values
    data = pd.DataFrame({
        'Gender': [gender],
        'Senior Citizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'Tenure Months': [tenure_months],
        'Phone Service': [phone_service],
        'Multiple Lines': [multiple_lines],
        'Internet Service': [internet_service],
        'Online Security': [online_security],
        'Online Backup': [online_backup],
        'Device Protection': [device_protection],
        'Tech Support': [tech_support],
        'Streaming TV': [streaming_tv],
        'Streaming Movies': [streaming_movies],
        'Contract': [contract],
        'Paperless Billing': [paperless_billing],
        'Payment Method': [payment_method],
        'Monthly Charges': [monthly_charges],
        'Total Charges': [total_charges]
    })

     # Get the input features from the form
#    inputs = [request.form.get(feature) for feature in feature_order]

    # Create a DataFrame from the input data
   # df = pd.DataFrame([data])

    # Preprocess the input data
    df_transformed = preprocess_data(data)
    
    # Preprocess the data
    preprocessed_data = preprocessor.transform(df_transformed)
    
    # Make predictions using the trained model
    predictions = model.predict_proba(preprocessed_data)
    print(predictions)
    
     # Get the probability of churn and format it to two decimal digits
    probability_of_churn = round(predictions[0][1] * 100, 2)
    # Return the predictions to the user
    return render_template('index.html', predictions=probability_of_churn)
    

if __name__ == '__main__':
    app.run(debug=True)