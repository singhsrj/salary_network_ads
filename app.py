import streamlit as st
import pandas as pd
import pickle

# Load the saved Decision Tree model (pkl file)
with open('decision_tree_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Streamlit App Title
st.title("Salary Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Function to capture user input
def user_input_features():
    # Creating sliders for the user to input feature values
    age = st.sidebar.slider('age', 19, 60, 19)
    salary = st.sidebar.slider('salary', 15000, 150000, 15000)
  
    
    # Store input features in a DataFrame
    data = {'age': age,
            'salary': salary,
            }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Capture the input features from the user
input_df = user_input_features()

# Display the input features in the Streamlit app
st.subheader('User Input Features')
st.write(input_df)

# Make prediction with the loaded model
prediction = classifier.predict(input_df)
prediction_proba = classifier.predict_proba(input_df)

# Display the class names and corresponding prediction
st.subheader('Class Labels and their Index Number')
st.write(['Setosa', 'Versicolour', 'Virginica'])

# Show the predicted class and the prediction probability
st.subheader('Prediction')
st.write(['Setosa', 'Versicolour', 'Virginica'][prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)
