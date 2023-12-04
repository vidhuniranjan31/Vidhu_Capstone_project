from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
# Initialize an empty list for cleaned suggestions
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

# Chatbot route
@app.route("/")
def index():
    return render_template('chat.html')

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

    # Find most similar users
    target_user_similarity = cosine_similarity([target_user_data], user_data_sample[selected_columns])
    #1X32k dimension(1D array)
    target_user_index = np.argmax(target_user_similarity)
    # returns the  index of the highest cosine similarity value
    similar_users = np.argsort(user_similarity[target_user_index])[::-1]
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
    

if __name__ == '__main__':
    app.run()
