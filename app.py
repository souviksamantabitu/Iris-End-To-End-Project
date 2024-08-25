# import packages
import pandas as pd
import numpy as np
import pickle 
import streamlit as st

#load the model and preprocessior
with open("notebook/pipeline.pkl","rb") as file1:
    pre = pickle.load(file1)

#load the model
with open("notebook/model.pkl","rb") as file2:
    model=pickle.load(file2)
    
# preprocess the data
def predict_data(sep_ln,sep_wdth,pet_ln,pet_wdth):
    dct ={
        "sepal_length":[sep_ln],
        "sepal_width":[sep_wdth],
        "petal_length":[pet_ln],
        "petal_width":[pet_wdth]
    }
    xnew=pd.DataFrame(dct)
    xnew_pre = pre.transform(xnew)
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob=np.max(prob)
    return pred,max_prob

#Run Streamlite app
if __name__=='__main__':
    st.set_page_config(page_title="Iris Project Souvik")
    st.title("Iris Project - Souvik Samanta")
    st.subheader('please provide below inputs')
    #take input from the user
    sep_ln=st.number_input("sepal length :",min_value=0.00,step=0.01)
    sep_wdth=st.number_input("sepa width :",min_value=0.00,step=0.01)
    pet_ln=st.number_input("petal length :",min_value=0.00,step=0.01)
    pet_wdth=st.number_input("petal width :",min_value=0.00,step=0.01)
    #create predict button
    submit=st.button("predict")
    #if submit button pressed
    if submit:
        pred,max_prob=predict_data(sep_ln,sep_wdth,pet_ln,pet_wdth)
        st.subheader("Model Response:")
        st.subheader(f"Prediction {pred}")
        st.subheader(f"Probability {max_prob: .4f}")
        st.progress(max_prob)