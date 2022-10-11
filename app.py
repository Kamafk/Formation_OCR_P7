# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:41:17 2022

@author: Yoann
"""

import pandas as pd
import pickle
import streamlit as st 
import shap

#Load the saved model
pickle_in = open("pipeline.pkl","rb")
pipeline = pickle.load(pickle_in)

#General option
st.set_page_config( 
    page_title="Loan Prediction App",)

st.set_option('deprecation.showPyplotGlobalUse', False)


######################
#Main page layout, before prediction
######################

#Partie 1
st.title("Loan Default Prediction") 
st.subheader("Are you sure your loan applicant is surely going to   pay the loan back?💸""This machine learning app will help you to make a prediction to help you with your decision!")


#Partie 2
st.subheader("To predict default/ failure to pay back status, you need to follow the steps below:") 
st.markdown(""" 
1. Enter/choose the parameters that best descibe your applicant on the left side bar; 
2. Press the "Predict" button and wait for the result. 
""")

#Partie 3
st.subheader("Below you could find prediction result: ")


######################
#Sidebar layout
######################
st.sidebar.title("Loan Applicant Info") 
st.sidebar.write("Please choose parameters that descibe the applicant")

def user_input_features():
    CODE_GENDER = st.sidebar.text_input("CODE_GENDER")
    EXT_SOURCE_1 = st.sidebar.text_input("EXT_SOURCE_1")
    EXT_SOURCE_2 = st.sidebar.text_input("EXT_SOURCE_2")
    EXT_SOURCE_3 = st.sidebar.text_input("EXT_SOURCE_3")
    NAME_CONTRACT_TYPE_Cash_loans = st.sidebar.text_input("NAME_CONTRACT_TYPE_Cash_loans")
    NAME_EDUCATION_TYPE_Higher_education = st.sidebar.text_input("NAME_EDUCATION_TYPE_Higher_education")
    NAME_EDUCATION_TYPE_Secondary___secondary_special = st.sidebar.text_input("NAME_EDUCATION_TYPE_Secondary___secondary_special")
    OCCUPATION_TYPE_Drivers = st.sidebar.text_input("OCCUPATION_TYPE_Drivers")
    CC_AMT_BALANCE_MEAN = st.sidebar.text_input("CC_AMT_BALANCE_MEAN")
    CC_AMT_DRAWINGS_ATM_CURRENT_SUM = st.sidebar.text_input("CC_AMT_DRAWINGS_ATM_CURRENT_SUM")
    CC_AMT_DRAWINGS_CURRENT_MEAN = st.sidebar.text_input("CC_AMT_DRAWINGS_CURRENT_MEAN")
    CC_AMT_RECEIVABLE_PRINCIPAL_MEAN = st.sidebar.text_input("CC_AMT_RECEIVABLE_PRINCIPAL_MEAN")
    CC_AMT_RECIVABLE_MEAN = st.sidebar.text_input("CC_AMT_RECIVABLE_MEAN")
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN = st.sidebar.text_input("CC_CNT_DRAWINGS_ATM_CURRENT_MEAN")
    CC_CNT_DRAWINGS_ATM_CURRENT_VAR = st.sidebar.text_input("CC_CNT_DRAWINGS_ATM_CURRENT_VAR")
    INSTAL_DPD_MEAN = st.sidebar.text_input("INSTAL_DPD_MEAN")
    APPROVED_AMT_DOWN_PAYMENT_MAX = st.sidebar.text_input("APPROVED_AMT_DOWN_PAYMENT_MAX")
    PREV_CODE_REJECT_REASON_XAP_MEAN = st.sidebar.text_input("PREV_CODE_REJECT_REASON_XAP_MEAN")
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN = st.sidebar.text_input("PREV_NAME_CONTRACT_STATUS_Refused_MEAN")
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN = st.sidebar.text_input("PREV_NAME_PRODUCT_TYPE_walk_in_MEAN")
    data = {'CODE_GENDER': CODE_GENDER,
            'EXT_SOURCE_1': EXT_SOURCE_1,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3,
            'NAME_CONTRACT_TYPE_Cash_loans': NAME_CONTRACT_TYPE_Cash_loans,
            'NAME_EDUCATION_TYPE_Higher_education': NAME_EDUCATION_TYPE_Higher_education,
            'NAME_EDUCATION_TYPE_Secondary___secondary_special': NAME_EDUCATION_TYPE_Secondary___secondary_special,
            'OCCUPATION_TYPE_Drivers': OCCUPATION_TYPE_Drivers,
            'CC_AMT_BALANCE_MEAN': CC_AMT_BALANCE_MEAN,
            'CC_AMT_DRAWINGS_ATM_CURRENT_SUM': CC_AMT_DRAWINGS_ATM_CURRENT_SUM,
            'CC_AMT_DRAWINGS_CURRENT_MEAN': CC_AMT_DRAWINGS_CURRENT_MEAN,
            'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN': CC_AMT_RECEIVABLE_PRINCIPAL_MEAN,
            'CC_AMT_RECIVABLE_MEAN': CC_AMT_RECIVABLE_MEAN,
            'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN': CC_CNT_DRAWINGS_ATM_CURRENT_MEAN,
            'CC_CNT_DRAWINGS_ATM_CURRENT_VAR': CC_CNT_DRAWINGS_ATM_CURRENT_VAR,
            'INSTAL_DPD_MEAN': INSTAL_DPD_MEAN,
            'APPROVED_AMT_DOWN_PAYMENT_MAX': APPROVED_AMT_DOWN_PAYMENT_MAX,
            'PREV_CODE_REJECT_REASON_XAP_MEAN': PREV_CODE_REJECT_REASON_XAP_MEAN,
            'PREV_NAME_CONTRACT_STATUS_Refused_MEAN': PREV_NAME_CONTRACT_STATUS_Refused_MEAN,
            'PREV_NAME_PRODUCT_TYPE_walk_in_MEAN': PREV_NAME_PRODUCT_TYPE_walk_in_MEAN,}
    user_features = pd.DataFrame(data, index=[0])
    return user_features

D_user_data = user_input_features()

######################
#Predict button, after prediction
######################

btn_predict = st.sidebar.button("Predict")

#Prediction
if btn_predict: 
    pred = pipeline.predict_proba(D_user_data)[:, 1]
    
    if pred[0] > 0.2: 
        st.error('The customer default probability is {}'.format(pred))           
    else: 
        st.success('The customer default probability is {}'.format(pred))
        
#SHAP graphique      
    st.subheader('Result Interpretability - Applicant Level')
    shap.initjs()
    explainer = shap.Explainer(pipeline["classifier"])
    shap_values = explainer(D_user_data)
    fig  = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values[0].values[:,0], feature_names = D_user_data.columns, max_display = 20 )
    st.pyplot(fig)