import streamlit as st
import pandas as pd
import pickle
import numpy as np
import utils as ut

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

dt_model = load_model('dt_model.pkl')
feat_voting_clf = load_model('feat_voting_clf.pkl')
feat_xgb_model = load_model('feat_xgb_model.pkl')
knn_model = load_model('knn_model.pkl')
lgb_model = load_model('lgb_model.pkl')
nb_model = load_model('nb_model.pkl')
rf_model = load_model('rf_model.pkl') 
smote_xgb_model = load_model('smote_xgb_model.pkl')
svm_model = load_model('svm_model.pkl')
voting_clf = load_model('voting_clf.pkl')
xgb_model = load_model('xgb_model.pkl')
SMOTE_rf_model = load_model('SMOTE_feat_rf_mode.pkl')
SMOTE_feat_svm_model = load_model('SMOTE_feat_svm_model.pkl')
hypertune_lgbmc_clf_model = load_model('hypertune_lgbmc_clf.pkl')
def prepare_input(credit_score, location, gender, age, tenure, balance, 
                  num_products, has_credit_card, is_active_member, estimated_salary, clv, tenure_age_ratio, age_group_middle_age, age_group_senior, age_group_elderly):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'CLV': clv,
        'TenureAgeRatio': tenure_age_ratio,
        'AgeGroup_MiddleAge': 1 if age_group_middle_age else 0,
        'AgeGroup_Senior': 1 if age_group_senior else 0,
        'AgeGroup_Elderly': 1 if age_group_elderly else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    probabilities = {
        # 'Decision Tree': dt_model.predict_proba(input_df)[0][1],
        # 'Feat Voting Classifier': feat_voting_clf.predict_proba(input_df)[0][1],
        'Feat XGBoost': feat_xgb_model.predict_proba(input_df)[0][1],
        # 'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
        # 'LightGBM': lgb_model.predict_proba(input_df)[0][1],
        # 'Naive Bayes': nb_model.predict_proba(input_df)[0][1],
        # 'Random Forest': rf_model.predict_proba(input_df)[0][1],
        'SMOTE XGBoost': smote_xgb_model.predict_proba(input_df)[0][1],
        'SMOTE RF': SMOTE_rf_model.predict_proba(input_df)[0][1],
        'SMOTE SVM': SMOTE_feat_svm_model.predict_proba(input_df)[0][1],
        # 'Hyper LGBMC': hypertune_lgbmc_clf_model.predict_proba(input_df)[0][1],
        # 'SVM': svm_model.predict_proba(input_df)[0][1],
        # 'Voting Classifier': voting_clf.predict_proba(input_df)[0][1],
        # 'XGBoost': xgb_model.predict_proba(input_df)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability



st.title("Customer Churn Predictions")

df = pd.read_csv("updated_churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _ , row in df.iterrows()]

selected_customer_option = st.selectbox("Select Customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split('-')[0])

    print("Selected Customer ID", selected_customer_id)

    selected_customer_surname = (selected_customer_option.split('-')[1])

    print("Selected Customer Surname", selected_customer_surname)

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id]

    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)


    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore'].iloc[0])) 

        location = st.selectbox(
            "Location", ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(selected_customer['Geography'].iloc[0]))

        gender = st.radio("Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'].iloc[0] == 'Male' else 1)

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(selected_customer['Age'].iloc[0]))

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure'].iloc[0]))


    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=float(selected_customer['Balance'].iloc[0]))

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts'].iloc[0]))

        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer['HasCrCard'].iloc[0]))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember'].iloc[0]))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary'].iloc[0]))
        
    clv = float(selected_customer['CLV'].iloc[0])
    tenure_age_ratio = float(selected_customer['TenureAgeRatio'].iloc[0])
    age_group_middle_age = bool(selected_customer['AgeGroup_MiddleAge'].iloc[0])
    age_group_senior = bool(selected_customer['AgeGroup_Senior'].iloc[0])
    age_group_elderly = bool(selected_customer['AgeGroup_Elderly'].iloc[0])

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary, clv, tenure_age_ratio, age_group_middle_age, age_group_senior, age_group_elderly)

    exited = bool(selected_customer['Exited'].iloc[0])

    avg_probability = make_predictions(input_df, input_dict)

    st.write(f"**Exited:** {'True' if exited else 'False'}")
