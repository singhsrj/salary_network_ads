import numpy as np
import pandas as pd


dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"""Spliting dataset into training set and testing set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)

"""Feature Scale the Estimated salary"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(criterion ='log_loss', random_state=0)

classifier.fit(X_train,y_train)


"""Prediction for whole y_test"""

#here , we concatenate the prediction with
#our testing set, to check whether the prediction is right or not

st.title("whether customer will buy or not")

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
# prediction_proba = classifier.predict_proba(input_df)

# # Display the class names and corresponding prediction
# st.subheader('Class Labels and their Index Number')
# st.write(['Setosa', 'Versicolour', 'Virginica'])

# Show the predicted class and the prediction probability
st.subheader('Prediction')
if(prediction==1) st.write("Yes")
else st.write("NO")
