from utils import columns

import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('xgb.joblib')
st.title('Did they survive? :ship:')
#Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
pclass = st.selectbox("Choose class ", [1,2,3])
name = st.text_input('Enter your name ', 'John Smith')
sex = st.select_slider("Choose your sex ",['male','female'])
age = st.slider('Choose an age ',0,100)
sibsp = st.slider('Number of siblings and spouse', 0, 10)
parch = st.slider('Number of parents and children', 0, 10)
ticket = st.text_input('Enter ticket number','12345')
fare = st.number_input('Choose your fare', 0, 1000)
cabin = st.text_input('Enter cabin details','C123')
embarked = st.selectbox('Choose where you were embarked ',['S','C','Q'])

def predict():
    X = pd.DataFrame([[pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]], columns = columns)
    y = model.predict(X)
    if y[0] == 1:
        st.success('Passenger survived :thumbsup:')
    else:
        st.error('Passenger drowned :thumbsdown:')

trigger = st.button('Lets Predict', on_click=predict)