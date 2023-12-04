from flask import Flask, url_for, render_template, request, redirect, session, flash,jsonify
from flask_sqlalchemy import SQLAlchemy
import sqlite3
import numpy as np
import pickle
import os
from cryptography.fernet import Fernet
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import secrets
# Create a StandardScaler object
# StSc = StandardScaler()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model1 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

#key = os.environ.get("MY_APP_ENCRYPTION_KEY")
key='32393945764575524e556e72617933674b5158794e59546d45635f39324e784132664d5553667a2d706d6b3d'
# print(key)

if key is None:
    print("hello")
    # If it doesn't exist, generate a new key
    key = Fernet.generate_key()
    print(key)
    # Convert the bytes key to a hexadecimal string and store it in an environment variable
    os.environ["MY_APP_ENCRYPTION_KEY"] = key.hex()
    print(os.environ["MY_APP_ENCRYPTION_KEY"])
else:
    # If the environment variable exists, convert it back to bytes for Fernet
    key = bytes.fromhex(key)

cipher_suite = Fernet(key)

def encrypt_data(data):
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data):
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    return decrypted_data.decode()

import joblib



app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
modelloan = joblib.load('modelloan.pkl')
StSc = joblib.load('fittedScaler.pkl')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = secrets.token_hex(16)
db = SQLAlchemy(app)



class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    # password = db.Column(db.String(100))
    password = db.Column(db.LargeBinary)

    def __init__(self, username, password):
        self.username = username
        self.password = password



class FormDetails(db.Model):
    __tablename__ = 'form_details'
    id = db.Column(db.Integer, primary_key=True)
    Age = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Annual_Income = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Monthly_Inhand_Salary = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Num_Bank_Accounts = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Delay_from_due_date = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Num_Credit_Card = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Num_of_Loan = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Num_of_Delayed_Payment = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Interest_Rate = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Changed_Credit_Limit = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Outstanding_Debt = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Credit_Utilization_Ratio = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Credit_History_Age = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Total_EMI_per_month = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Amount_invested_monthly = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Monthly_Balance = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Num_Credit_Inquiries = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Credit_Mix = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data
    Payment_of_Min_Amount = db.Column(db.LargeBinary)  # Use LargeBinary for encrypted data


class UserFormId(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    form_id = db.Column(db.Integer)
    credit_score = db.Column(db.Float)
def create_database():
    database_file = 'users.db'
    with app.app_context():
        conn = sqlite3.connect(database_file)
        conn.close()

# Function to create the database tables
def create_tables():
    with app.app_context():
        db.create_all()


#####################################################################################

cleaned_suggestions = []
introduction_questions = [
    "Hello! I'm your friendly chatbot Cibi. What can I help you with today?",
    "Would you like to know something specific or just have a casual chat?",
]
# Load the user data and suggestions data
file_path = 'train.csv'  # Update with your file path
user_data = pd.read_csv(file_path)

# Relevant columns for collaborative filtering
selected_columns = [
    "Num_Credit_Card", "Interest_Rate" ,
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt"
]

# Set a smaller sample size
sample_size =32000  # Set a smaller sample size

if len(user_data) > sample_size:
    sample_indices = np.random.choice(len(user_data), size=sample_size, replace=False)
    user_data_sample = user_data.iloc[sample_indices]
else:
    user_data_sample = user_data

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_data_sample[selected_columns])
#dimension of this is 30kX30k

# Read suggestions from JSON file
with open("suggestions.json", "r") as json_file:
    suggestions_data = json.load(json_file)

suggestions = {item["name"]: item["feature_indices"] for item in suggestions_data["suggestions"] if "Credit_Score" not in item["feature_indices"]}


# Define a function to check if a suggestion is relevant for the customer
def is_relevant_suggestion(user_average, similar_users_average, threshold):
    difference = np.abs(user_average - np.mean(similar_users_average))
    print(f"User Average: {user_average}, Similar Users Average: {np.mean(similar_users_average)}, Difference: {difference}, Threshold: {threshold}")
    return difference > threshold


@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return render_template('home.html')
       
    else:
        return render_template('index.html', message="Hello!")

@app.route("/get", methods=["POST"])
def chat():
    user_params = request.form.get("userParams")
    print("hello")
    print(user_params)
    if not user_params:
        return jsonify(["Please enter more parameters to get suggestions."])

    user_params = [float(param) for param in user_params.split(',')]
    
    # Convert user input to a NumPy array
    target_user_data = np.array(user_params)
    print("hello2")
    # Find most similar users
    target_user_similarity = cosine_similarity([target_user_data], user_data_sample[selected_columns])
    target_user_index = np.argmax(target_user_similarity)
    similar_users = np.argsort(user_similarity[target_user_index])[::-1]
    print("hello3")
    similar_users = similar_users[similar_users != target_user_index]  # Exclude the target user

    # Print credit score improvement suggestions for the user
    relevant_suggestions = []
    for suggestion, feature_indices in suggestions.items():
        user_average = np.mean(target_user_data[feature_indices])
        similar_users_average = np.mean(user_data_sample.iloc[similar_users, feature_indices], axis=1)
        # Define a threshold for each suggestion (customize as needed)
        if is_relevant_suggestion(user_average, similar_users_average, threshold= 4):
            relevant_suggestions.append(suggestion)
            relevant_suggestions = relevant_suggestions[:7]

        # json_responses = []
        formatted_suggestions = []

# # Convert each relevant suggestion to JSON and add it to the list
        for suggestion in relevant_suggestions:
            formatted_suggestions.append(str(suggestion))

# # Return the JSON responses as a list
    return jsonify(formatted_suggestions)
    
###################################################################################################

@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            password_value=request.form['password']
            encrypted_password = encrypt_data(password_value)
            db.session.add(User(username=request.form['username'], password=encrypted_password))
            db.session.commit()
            return redirect(url_for('login'))
        except:
            return render_template('index.html', message="User Already Exists")
    else:
        return render_template('register.html')


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        
        data = User.query.filter_by(username=u).first()
        decrypted_password = decrypt_data(data.password)
        if (data is not None) and (decrypted_password==p):
            # decrypted_password = decrypt_data(data.password)
            # if decrypted_password==p:
            session['logged_in'] = True
            session['user_id'] = data.id
            return redirect(url_for('index'))
            # return render_template('index.html', message="Incorrect Details")
        return render_template('index.html', message="Incorrect Details")


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))


@app.route('/loan', methods=['GET', 'POST'])
def loanapproval():
    if request.method == 'POST':
        print("hi1")
        coapplicant_income = float(request.form['CoapplicantIncome'])
        if coapplicant_income == 0:
            coapplicant_income_log = 0
            print("helloinsideif")
        else:
            coapplicant_income_log = np.log(coapplicant_income)
        print("hi1.5")
        try:
            print("hi1.7")
            user_input_values = [
    np.log(float(request.form.get('ApplicantIncome'))),
    coapplicant_income_log,
    np.log(float(request.form.get('LoanAmount'))),
    float(request.form.get('Loan_Amount_Term')),
    request.form.get('Credit_History','No'),  # Provide a default value
    request.form.get('Gender', 'Male'),  # Provide a default value
    request.form.get('Married', 'No'),  # Provide a default value
    request.form.get('Dependents', '0'),  # Provide a default value
    request.form.get('Education', 'Not Graduate'),  # Provide a default value
    request.form.get('Self_Employed', 'No'),  # Provide a default value
    request.form.get('Property_Area')  # Provide a default value
]
            
         
            print("hi2")
            gender_mapping = {'Male': 1, 'Female': 0}
            credithistory_mapping = {'Yes': 1, 'No': 0}
            married_mapping = {'Yes': 1, 'No': 0}
            dependents_mapping = {'3+': 3, '1': 1, '2': 2, '0': 0}
            education_mapping = {'Graduate': 1, 'Not Graduate': 0}
            self_employed_mapping = {'Yes': 1, 'No': 0}
            property_area_mapping = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}

            user_input_values[4] = credithistory_mapping.get(user_input_values[4])
            user_input_values[5] = gender_mapping.get(user_input_values[5])
            user_input_values[6] = married_mapping.get(user_input_values[6])
            user_input_values[7] = dependents_mapping.get(user_input_values[7])
            user_input_values[8] = education_mapping.get(user_input_values[8])
            user_input_values[9] = self_employed_mapping.get(user_input_values[9])
            user_input_values[10] = property_area_mapping.get(user_input_values[10])
            print(user_input_values)
            print("hi3")
            user_input_array = np.array(user_input_values).reshape(1, -1)
            print("hi4")
        
            scaled_user_input = StSc.transform(user_input_array)
            prediction_value = modelloan.predict(scaled_user_input)
            print("hi5")
            print(prediction_value)
            prediction_result = int(prediction_value[0])
            return render_template('loanform.html', prediction=prediction_result)
        except Exception as e:
            print("An error occurred:", e)
            return jsonify({'error': 'An error occurred while processing the request'})

    else:
        return render_template('loanform.html')



@app.route('/predict', methods=['GET', 'POST'])
def predictscore():
    if request.method == 'POST':
        try:
            age = (request.form['Age'])
            print(age)
            annual_income = (request.form.get('Annual_Income'))
            num_bank_accounts = (request.form.get('Num_Bank_Accounts'))
            delay_from_due_date = (request.form.get('Delay_from_due_date'))
            num_credit_card = (request.form.get('Num_Credit_Card'))
            num_of_loan = (request.form.get('Num_of_Loan'))
            num_of_delayed_payment = (request.form.get('Num_of_Delayed_Payment'))
            interest_rate = (request.form.get('Interest_Rate'))
            changed_credit_limit = (request.form.get('Changed_Credit_Limit'))
            outstanding_debt = (request.form.get('Outstanding_Debt'))
            credit_utilization_ratio = (request.form.get('Credit_Utilization_Ratio'))
            credit_history_age = (request.form.get('Credit_History_Age'))
            total_emi_per_month = (request.form.get('Total_EMI_per_month'))
            amount_invested_monthly = (request.form.get('Amount_invested_monthly'))
            monthly_balance = (request.form.get('Monthly_Balance'))
            num_credit_inquiries = (request.form.get('Num_Credit_Inquiries'))
            monthly_inhand_salary = (request.form.get('Monthly_Inhand_Salary'))
            credit_mix = request.form['Credit_Mix']
            payment_of_min_amount = request.form['Payment_of_Min_Amount']
          


            Credit_Mix_Good = 0
            Credit_Mix_Bad = 0
            Credit_Mix_Standard = 0

            # Check the value of credit_mix and set the corresponding variable to 1
            if credit_mix == 'good':
                Credit_Mix_Good = 1
            elif credit_mix == 'bad':
                Credit_Mix_Bad = 1
            elif credit_mix == 'standard':
                Credit_Mix_Standard = 1

            Payment_of_Min_Amount_No = 0
            Payment_of_Min_Amount_Yes = 0
            Payment_of_Min_Amount_NM = 0

            if payment_of_min_amount == 'No':
                Payment_of_Min_Amount_No = 1
            elif payment_of_min_amount == 'Yes':
                Payment_of_Min_Amount_Yes = 1



            model_input_array = np.array([
            age, annual_income,monthly_inhand_salary, num_bank_accounts,num_credit_card, interest_rate,  num_of_loan,delay_from_due_date,
             num_of_delayed_payment, changed_credit_limit,num_credit_inquiries, outstanding_debt, credit_utilization_ratio,
            credit_history_age, total_emi_per_month, amount_invested_monthly,
            monthly_balance,Credit_Mix_Bad,Credit_Mix_Good,Credit_Mix_Standard,Payment_of_Min_Amount_NM,
            Payment_of_Min_Amount_No,Payment_of_Min_Amount_Yes])

            
            print(model_input_array)
            model_input_array = model_input_array.reshape(1, -1)
            prediction = model.predict(model_input_array)
            print(prediction)

            encrypted_age = encrypt_data(age)
            encrypted_annual_income = encrypt_data(annual_income)
            encrypted_monthly_inhand_salary = encrypt_data(monthly_inhand_salary)
            encrypted_num_bank_accounts = encrypt_data(num_bank_accounts)
            encrypted_delay_from_due_date = encrypt_data(delay_from_due_date)
            encrypted_num_credit_card = encrypt_data(num_credit_card)
            encrypted_num_of_loan = encrypt_data(num_of_loan)
            encrypted_num_of_delayed_payment = encrypt_data(num_of_delayed_payment)
            encrypted_interest_rate = encrypt_data(interest_rate)
            encrypted_changed_credit_limit = encrypt_data(changed_credit_limit)
            encrypted_outstanding_debt = encrypt_data(outstanding_debt)
            encrypted_credit_utilization_ratio = encrypt_data(credit_utilization_ratio)
            encrypted_credit_history_age = encrypt_data(credit_history_age)
            encrypted_total_emi_per_month = encrypt_data(total_emi_per_month)
            encrypted_amount_invested_monthly = encrypt_data(amount_invested_monthly)
            encrypted_monthly_balance = encrypt_data(monthly_balance)
            encrypted_num_credit_inquiries = encrypt_data(num_credit_inquiries)
            encrypted_credit_mix = encrypt_data(credit_mix)
            encrypted_payment_of_min_amount = encrypt_data(payment_of_min_amount)

            new_data = FormDetails(
            Age=encrypted_age,
            Annual_Income=encrypted_annual_income,
            Monthly_Inhand_Salary=encrypted_monthly_inhand_salary,
            Num_Bank_Accounts=encrypted_num_bank_accounts,
            Delay_from_due_date=encrypted_delay_from_due_date,
            Num_Credit_Card=encrypted_num_credit_card,
            Num_of_Loan=encrypted_num_of_loan,
            Num_of_Delayed_Payment=encrypted_num_of_delayed_payment,
            Interest_Rate=encrypted_interest_rate,
            Changed_Credit_Limit=encrypted_changed_credit_limit,
            Outstanding_Debt=encrypted_outstanding_debt,
            Credit_Utilization_Ratio=encrypted_credit_utilization_ratio,
            Credit_History_Age=encrypted_credit_history_age,
            Total_EMI_per_month=encrypted_total_emi_per_month,
            Amount_invested_monthly=encrypted_amount_invested_monthly,
            Monthly_Balance=encrypted_monthly_balance,
            Num_Credit_Inquiries=encrypted_num_credit_inquiries,
            Credit_Mix=encrypted_credit_mix,
            Payment_of_Min_Amount=encrypted_payment_of_min_amount
        )
        
            
            db.session.add(new_data)
            db.session.commit()
            formId = new_data.id
            userId = session.get('user_id')
            db.session.add(UserFormId(user_id=userId,form_id=formId,credit_score=prediction))
            db.session.commit()    
        except Exception as e: 
            flash("An error occurred while storing data. Please try again.", e)
        return render_template('form.html', prediction=int(prediction))
            
    else:
        return render_template('form.html')



    

@app.route('/cibilchatbot', methods=['GET'])
def cibilchatbot():
    return render_template('chat.html')



if(__name__ == '__main__'):
    app.secret_key = "ThisIsNotASecret:p"
        # Create the SQLite database
    create_database()
    
    # Create the database tables
    create_tables()
    # db.create_all()
    app.run()
