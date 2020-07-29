import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')
# print(df['Outcome'].values)
# print(df.describe(include='all'))
# print(df.info())
st.header('Diabetes Prediction ML Model')
st.subheader('This Machine Learning WebApp can predict Diabetes based on provided values')
st.text('Data Table: Based on we built this model ')
# st.dataframe(df)
st.dataframe(df.style.highlight_max(axis=0))
chart = st.bar_chart(df)

# Now splitting data into text set and train set
# defining independent dataset
X = df.iloc[:, 0:8].values
# dependent dataset
Y = df.iloc[:, -1].values

# Splitting datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=30)

print(type(Y))

# get user input for future prediction
def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 17, 3)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 31)
    BMI = st.sidebar.slider('BMI', 0.0, 68.0, 27.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.45, 0.3725)
    Age= st.sidebar.slider('Age', 21, 91, 29)

    # dictionary that hold user input
    input_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPresure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insuline': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    # converting dictionary to dataframe
    userdata = pd.DataFrame(input_data, index=[0])
    return userdata
# storing user_input into a variable
user_input = get_user_input()

# displaying userinput in webapp
st.subheader('User Input: ')
st.write(user_input)

# ML Model
Prediction_Model = RandomForestClassifier()
Prediction_Model.fit(X_train, Y_train)

st.subheader('Model Test Accuracy: ')
accuracy = str(accuracy_score(Y_test, Prediction_Model.predict(X_test))*100)+'%'
st.write(accuracy)

# now predicting user input and Displaying it
prediction = Prediction_Model.predict(user_input)
st.subheader('Prediction')
st.text('Based on User Input')
if int(prediction) == 1:
    pred = f'There is {accuracy} chance that you have Diabetes!'
    st.write(pred)
else:
    pred = f'There is {accuracy} chance that you are Healthy, Awesome!'
    st.write(pred)
st.subheader('Classification: ')
st.write(prediction)

# hide menu and footer
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
