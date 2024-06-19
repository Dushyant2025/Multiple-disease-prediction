# -*- coding: utf-8 -*-
"""frrferf
"""
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
from PIL import Image
import tensorflow as tf

# streamlit run "C:/Users/GOPAL/OneDrive/Desktop/multiple disease/hehe.py"
MODEL_PATH = "C:/Users/GOPAL/OneDrive/Desktop/multiple disease/braintumor.h5"  
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

# model1=tf.keras.models.load_model('C:/Users/GOPAL/OneDrive/Desktop/multiple disease/lungs.h5')
# class_labels = ['Normal', 'Lung Opacity','Viral Pneumonia']

diabetes_model = pickle.load(open("C:/Users/GOPAL/OneDrive/Desktop/multiple disease/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("C:/Users/GOPAL/OneDrive/Desktop/multiple disease/heart_model.sav", "rb"))

# Create a sidebar
with st.sidebar:
    # Use option_menu for selecting disease
    selected = option_menu('Multiple disease prediction system',
                           
                           ['Diabetes Prediction',
                            'Heart Prediction',
                            'Brain Disease'
                            ],
                            
                           icons=['activity','heart-pulse',''],
                           default_index=0)

# Display appropriate form based on selection
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')
    

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies',placeholder="put 0 if male")

    with col2:
        Glucose = st.text_input('Glucose Level',placeholder="in mg/dl")

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)



elif selected == 'Heart Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
    
elif selected=="Brain Disease":
    
    st.title("Brain Disease Detection")
    st.write("Upload an MRI image for brain disease detection.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Perform prediction when the "Predict" button is clicked
        if st.button("Predict"):
            # Preprocess the image and make prediction
            image = np.array(image.resize((150, 150)))   # Resize and normalize the image
           
            
            image = image.reshape(1,150,150,3)
            predictions = model.predict(image)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions)
            

            # Display prediction result
            st.success(f"Predicted Class: {predicted_class}")
            st.success(f"Confidence: {confidence:.2f}")

