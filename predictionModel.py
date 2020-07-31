import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')
# print(df['Outcome'].values)
# print(df.describe(include='all'))
# print(df.info())
st.header('Prediction Model ML-WebApp')
st.subheader('This Machine Learning WebApp can predict Diabetes based on input values')
st.text('Data Table: Based on we built this model ')


classification = st.sidebar.selectbox("Select Classifier: ", ("Random Forest", "SVM", "KNN"))


# st.dataframe(df)
st.dataframe(df.style.highlight_max(axis=0))
st.write("Shape of Dataset: ", df.shape)
st.subheader('Dataset Statistics: ')
st.write(df.describe(include='all'))
chart = st.bar_chart(df)


# Now splitting data into text set and train set
# defining independent dataset
X = df.iloc[:, 0:8].values  # all rows of 0 to 8-1 = 7 columns
# dependent dataset
Y = df.iloc[:, -1].values  # all rows of very last column

# Splitting datasets
# st.sidebar.text('Random State')
random_state = st.sidebar.slider('Random State: ', 7, 30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=random_state)


def model_param(cls_name):
    param = dict()
    if cls_name == 'KNN':
        k = st.sidebar.slider('K: ', 1, 15)
        param['k'] = k
    elif cls_name == 'SVM':
        c = st.sidebar.slider('C: ', 0.1, 10.0)
        param['c'] = c
    else:
        max_depth = st.sidebar.slider('Number of depth: ', 2, 15)
        n_estimators = st.sidebar.slider('Number of Estimator: ', 1, 100)
        param['Number of depth'] = max_depth
        param['Number of Estimator'] = n_estimators
    return param


model_param(classification)


# get user input for future prediction
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 17, 3)
    insulin = st.sidebar.slider('Insulin', 0, 846, 31)
    bmi = st.sidebar.slider('BMI', 0.0, 68.0, 27.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.45, 0.3725)
    age = st.sidebar.slider('Age', 21, 91, 29)

    # dictionary that hold user input
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    # converting dictionary to dataframe
    user_data = pd.DataFrame(input_data, index=[0])
    return user_data


# storing user_input into a variable
user_input = get_user_input()


# displaying userinput in webapp
st.subheader('User Input: ')
st.write(user_input)


# ML Model
Prediction_Model = RandomForestClassifier()
Prediction_Model.fit(X_train, Y_train)

# Prediction
prediction = Prediction_Model.predict(X_test)
st.subheader('Test Accuracy: ')
accuracy = str(accuracy_score(Y_test, prediction)*100)+'%'
st.write(accuracy)

# now predicting user input and Displaying it
prediction = Prediction_Model.predict(user_input)
st.subheader('Prediction: ')
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