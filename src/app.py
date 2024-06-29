import os
import pickle
import streamlit as st

#Load the model
with open('/workspaces/machine-learning-Streamlit-Juli-MM/models/tree_classifier_crit-gini_maxdepth-5_minleaf-2_minsplit-10_42.sav', 'rb') as file:
    model = pickle.load(file)

# Streamlit app

    st.title("Diabetes Predictor")
    
    st.write("This predictor uses a Decision Tree optimized model with a 80% accuracy in predicting, based on diagnostic measures, whether or not a patient has diabetes.")
    st.write("Please fill in the blanks with your information:")
    
    number1 = st.number_input("Pregnancies", value=0, min_value=0, step=1)
    st.divider()
    number2 = st.number_input("Glucose", value=0, min_value=0, step=1)
    st.divider()
    number3 = st.number_input("BloodPressure", value=0, min_value=0, step=1)
    st.divider()
    number4 = st.number_input("BMI", value=0.0, min_value=0.0, format="%.2f")
    st.divider()
    number5 = st.number_input("Diabetes Pedigree Function", value=0.0, min_value=0.0, format="%.3f")
    st.divider()
    number6 = st.number_input("Age", value=0, min_value=0, step=1)
    
    if st.button("Predict"):
        try:
            prediction = model.predict([[number1, number2, number3, number4, number5, number6]])[0]
            class_dict = {0: "Non diabetic", 1: "Diabetic"}
            pred_class = class_dict[prediction]
            st.write("Prediction:", pred_class)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

