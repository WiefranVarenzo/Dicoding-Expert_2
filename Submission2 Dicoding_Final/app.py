import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('model/dropout_prediction_model_ensemble.joblib')
encoder = joblib.load('model/encoder.joblib')

# Define the app layout
st.image("Banner.png", width=700)
st.header('Dropout Prediction App (Prototype)')

data = pd.DataFrame()

# Organize input fields in a logical and clear layout
st.subheader("Curriculum Information")

# Curriculum 2nd Semester Inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    Curricular_units_2nd_sem_enrolled = int(st.number_input(label='Curricular Units 2nd Sem Enrolled', value=0, min_value=0))
    data["Curricular_units_2nd_sem_enrolled"] = [Curricular_units_2nd_sem_enrolled]

with col2:
    Curricular_units_2nd_sem_approved = int(st.number_input(label='Curricular Units 2nd Sem Approved', value=0, min_value=0))
    data["Curricular_units_2nd_sem_approved"] = [Curricular_units_2nd_sem_approved]

with col3:
    Curricular_units_2nd_sem_evaluations = int(st.number_input(label='Curricular Units 2nd Sem Evaluations', value=0, min_value=0))
    data["Curricular_units_2nd_sem_evaluations"] = [Curricular_units_2nd_sem_evaluations]

with col4:
    Curricular_units_2nd_sem_grade = float(st.number_input(label='Curricular Units 2nd Sem Grade', value=0.0, min_value=0.0))
    data["Curricular_units_2nd_sem_grade"] = [Curricular_units_2nd_sem_grade]

# Curriculum 1st Semester Inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    Curricular_units_1st_sem_enrolled = int(st.number_input(label='Curricular Units 1st Sem Enrolled', value=0, min_value=0))
    data["Curricular_units_1st_sem_enrolled"] = [Curricular_units_1st_sem_enrolled]

with col2:
    Curricular_units_1st_sem_approved = int(st.number_input(label='Curricular Units 1st Sem Approved', value=0, min_value=0))
    data["Curricular_units_1st_sem_approved"] = [Curricular_units_1st_sem_approved]

with col3:
    Curricular_units_1st_sem_evaluations = int(st.number_input(label='Curricular Units 1st Sem Evaluations', value=0, min_value=0))
    data["Curricular_units_1st_sem_evaluations"] = [Curricular_units_1st_sem_evaluations]

with col4:
    Curricular_units_1st_sem_grade = float(st.number_input(label='Curricular Units 1st Sem Grade', value=0.0, min_value=0.0))
    data["Curricular_units_1st_sem_grade"] = [Curricular_units_1st_sem_grade]

st.subheader("Numerical Information")

# Numerical Inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    Admission_grade = float(st.number_input(label='Admission Grade', value=127.3, min_value=0.0))
    data["Admission_grade"] = [Admission_grade]

with col2:
    Age_at_enrollment = int(st.number_input(label='Age at Enrollment', value=20, min_value=0))
    data["Age_at_enrollment"] = [Age_at_enrollment]

with col3:
    Previous_qualification_grade = float(st.number_input(label='Previous Qualification Grade', value=0.0, min_value=0.0))
    data["Previous_qualification_grade"] = [Previous_qualification_grade]

with col4:
    GDP = float(st.number_input(label='GDP', value=0.0))
    data["GDP"] = [GDP]

st.subheader("Categorical Information")

# Categorical Inputs
col1, col2, col3 = st.columns(3)

with col1:
    Debtor = st.selectbox(label='Debtor', options=["yes", "no"], index=1)
    data["Debtor"] = [Debtor]

with col2:
    Scholarship_holder = st.selectbox(label='Scholarship Holder', options=["yes", "no"], index=1)
    data["Scholarship_holder"] = [Scholarship_holder]

with col3:
    Tuition_fees_up_to_date = st.selectbox(label='Tuition Fees Up to Date', options=["yes", "no"], index=1)
    data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]

# View raw data
with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800)

# Prediction
if st.button('Predict'):
    # Preprocess new data
    binary_features = ['Tuition_fees_up_to_date', 'Debtor', 'Scholarship_holder']
    for feature in binary_features:
        data[feature] = data[feature].map({'yes': 1, 'no': 0})

    categorical_features = data.select_dtypes(include=['object']).columns
    data_encoded = encoder.transform(data[categorical_features])
    data_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(categorical_features))

    new_data_processed = pd.concat([data.drop(columns=categorical_features).reset_index(drop=True), data_encoded], axis=1)

    # Additional checks to ensure proper preprocessing
    st.write("Preprocessed Data Sample:")
    st.write(new_data_processed.head())

    # Get the feature names used during model training
    trained_features = model.feature_names_in_
    features = new_data_processed[trained_features]

    # Ensure that the features match the trained features
    missing_features = set(trained_features) - set(features.columns)
    if missing_features:
        st.error(f"Missing features: {missing_features}")
    else:
        # Make prediction
        prediction_result = model.predict(features)
        prediction_proba = model.predict_proba(features)

        st.write("Dropout Prediction: {}".format(prediction_result[0]))
        st.write("Prediction Probabilities: {}".format(prediction_proba[0]))

        # Check if probabilities sum up to 1 (they should)
        if not (0.99 < prediction_proba[0].sum() < 1.01):
            st.error("Probabilities do not sum up to 1, something might be wrong.")

        # Provide more detailed debugging information
        st.write("Detailed Probability Distribution:")
        for class_name, proba in zip(model.classes_, prediction_proba[0]):
            st.write(f"{class_name}: {proba:.2f}")
