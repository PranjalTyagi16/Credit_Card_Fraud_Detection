import pickle
import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

loaded_model=pickle.load(open('credit_fraud.sav','rb'))

def credit_card_fraud(input_data):

    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'Legal Transaction'
    else:
        return 'Fraud'


def main():
    st.title('Credit Card Fraud Detection')

    col1,col2,col3,col4,col5=st.columns(5)

    with col1:
        Time = st.number_input('Enter the time')
    with col2:
        V1 = st.number_input('Enter the Value1')
    with col3:
        V2 = st.number_input('Enter the Value2')
    with col4:
        V3 = st.number_input('Enter the Value3')
    with col5:
        V4 = st.number_input('Enter the Value4')
    with col1:
        V5 = st.number_input('Enter the Value5')
    with col2:
        V6 = st.number_input('Enter the Value6')
    with col3:
        V7 = st.number_input('Enter the Value7')
    with col4:
        V8 = st.number_input('Enter the Value8')
    with col5:
        V9 = st.number_input('Enter the Value9')
    with col1:
        V10 = st.number_input('Enter the Value10')
    with col2:
        V11= st.number_input('Enter the Value11')
    with col3:
        V12= st.number_input('Enter the Value12')
    with col4:
        V13= st.number_input('Enter the Value13')
    with col5:
        V14= st.number_input('Enter the Value14')
    with col1:
        V15= st.number_input('Enter the Value15')
    with col2:
        V16= st.number_input('Enter the Value16')
    with col3:
        V17= st.number_input('Enter the Value17')
    with col4:
        V18= st.number_input('Enter the Value18')
    with col5:
        V19= st.number_input('Enter the Value19')
    with col1:
        V20= st.number_input('Enter the Value20')
    with col2:
        V21= st.number_input('Enter the Value21')
    with col3:
        V22= st.number_input('Enter the Value22')
    with col4:
        V23= st.number_input('Enter the Value23')
    with col5:
        V24= st.number_input('Enter the Value24')
    with col1:
        V25= st.number_input('Enter the Value25')
    with col2:
        V26= st.number_input('Enter the Value26')
    with col3:
        V27= st.number_input('Enter the Value27')
    with col4:
        V28= st.number_input('Enter the Value28')
    with col5:
        Amount = st.number_input('Enter the Amount')



    diagnosis=''
    if st.button('Fraud_Detection'):
        diagnosis=credit_card_fraud([Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount])

    st.success(diagnosis)


if __name__== '__main__':
    main()
