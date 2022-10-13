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
st.markdown("<h1 style='text-align: center; color: black;'> Prêt à dépenser : loan default prediction </h1>", unsafe_allow_html = True)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    st.image("Logo.png")

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
    CODE_GENDER = st.sidebar.radio('CODE_GENDER', ('0', '1'), index = 1)
    EXT_SOURCE_1 = st.sidebar.slider("EXT_SOURCE_1", min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.5)
    EXT_SOURCE_2 = st.sidebar.slider("EXT_SOURCE_2", min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.5)
    EXT_SOURCE_3 = st.sidebar.slider("EXT_SOURCE_3", min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.5)
    NAME_CONTRACT_TYPE_Cash_loans = st.sidebar.radio('NAME_CONTRACT_TYPE_Cash_loans', ('0', '1'), index = 1)
    NAME_EDUCATION_TYPE_Higher_education = st.sidebar.radio('NAME_EDUCATION_TYPE_Higher_education', ('0', '1'), index = 0)
    NAME_EDUCATION_TYPE_Secondary___secondary_special = st.sidebar.radio('NAME_EDUCATION_TYPE_Secondary___secondary_special', ('0', '1'), index = 1)
    OCCUPATION_TYPE_Drivers = st.sidebar.radio('OCCUPATION_TYPE_Drivers', ('0', '1'), index = 0)
    CC_AMT_BALANCE_MEAN = st.sidebar.number_input("CC_AMT_BALANCE_MEAN", min_value = 0, max_value = 400000, value = 0)
    CC_AMT_DRAWINGS_ATM_CURRENT_SUM = st.sidebar.number_input("CC_AMT_DRAWINGS_ATM_CURRENT_SUM", min_value = 0, max_value = 1500000, value = 0)
    CC_AMT_DRAWINGS_CURRENT_MEAN = st.sidebar.number_input("CC_AMT_DRAWINGS_CURRENT_MEAN", min_value = 0, max_value = 200000, value = 0)
    CC_AMT_RECEIVABLE_PRINCIPAL_MEAN = st.sidebar.number_input("CC_AMT_RECEIVABLE_PRINCIPAL_MEAN", min_value = 0, max_value = 400000, value = 0)
    CC_AMT_RECIVABLE_MEAN = st.sidebar.number_input("CC_AMT_RECIVABLE_MEAN", min_value = 0, max_value = 400000, value = 0)
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN = st.sidebar.slider("CC_CNT_DRAWINGS_ATM_CURRENT_MEAN", min_value = 0.0, max_value = 5.0, step = 0.5, value = 0.0)
    CC_CNT_DRAWINGS_ATM_CURRENT_VAR = st.sidebar.slider("CC_CNT_DRAWINGS_ATM_CURRENT_VAR", min_value = 0, max_value = 25, step = 1, value = 0)
    INSTAL_DPD_MEAN = st.sidebar.slider("INSTAL_DPD_MEAN", min_value = 0, max_value = 100, step = 1, value = 0)
    APPROVED_AMT_DOWN_PAYMENT_MAX = st.sidebar.number_input("APPROVED_AMT_DOWN_PAYMENT_MAX", min_value = 0, max_value = 300000, value = 0)
    PREV_CODE_REJECT_REASON_XAP_MEAN = st.sidebar.slider("PREV_CODE_REJECT_REASON_XAP_MEAN", min_value = 0.0, max_value = 1.0, step = 0.1, value = 1.0)
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN = st.sidebar.slider("PREV_NAME_CONTRACT_STATUS_Refused_MEAN", min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.0)
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN = st.sidebar.slider("PREV_NAME_PRODUCT_TYPE_walk_in_MEAN", min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.0)
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
    pred = pipeline.predict_proba(D_user_data)[0,1]
    
    if pred > 0.05: 
        st.error('The customer default probability is {:.2%}'.format(pred))           
    else: 
        st.success('The customer default probability is {:.2%}'.format(pred))
        
#Model default probability
    st.subheader('Probability distribution of client defects')
    st.image("Default_probability.png")
    st.write(""" If the probability is higher than 10%, please refuse the loan """)
    
#Model predictive quality
    st.subheader("Evaluation of the model's predictive quality")
    st.image("ROC_curve.png")

#SHAP graph 1 : Result Interpretability - Applicant Level     
    st.subheader('Result Interpretability - Applicant Level')
    shap.initjs()
    explainer = shap.TreeExplainer(pipeline["classifier"])
    shap_values = explainer(D_user_data)
    fig  = shap.plots.waterfall(shap_values[0], max_display = 20)
    st.pyplot(fig)
    
#SHAP graph 2 : Model Interpretability - Overall
    st.subheader('Model Interpretability - Overall')
    st.image("SHAP_summary.png")
    
    st.subheader('Interpretation:')      
    st.write(""" In this chart blue and red mean the feature value, blue is a smaller value and red is a higher value.
              \n The width of the bars represents the number of observations on a certain feature value, for example with the INSTALL_DPD_Mean feature we can see that most of the applicants are within the blue area.
              \n On axis x negative SHAP values represent applicants likely to pay the loan back and the positive values on the right side represent applicants that are that are likely to churn.
              \n What we are learning from this chart is that features such as EXT_Source_2 and EXT_Source_3 are the most impactful features driving the outcome prediction.
              \n The higher NAME_EDUCATION_TYPE_Higher_education is, the more likely the applicant to pay the loan back and vice versa.
             """)
