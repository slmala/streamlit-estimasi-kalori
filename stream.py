import pickle
import streamlit as st 
import setuptools
from PIL import Image


# membaca model
cal_model = pickle.load(open('estimasi_kalori.sav','rb'))
image = Image.open('banner.jpeg')

#judul web
st.image(image, caption='')
st.title('Aplikasi Prediksi Nilai Kalori')

col1, col2,col3=st.columns(3)
with col1:
    cal_fat = st.number_input('Input Nilai cal_fat :')
with col2:
    total_fat  = st.number_input('Input total_fat :')
with col3:
    sat_fat  = st.number_input('sat_fat :')
with col1:
    trans_fat = st.number_input('Input trans_fat :')
with col2:
    cholesterol = st.number_input('cholesterol :')
with col3:
    sodium = st.number_input('sodium :')
with col1:
    total_carb = st.number_input('total_carb :')
with col2:
    protein = st.number_input('protein :')

#code untuk estimasi
ins_est=''

#membuat button
with col1:
    if st.button('Estimasi Kalori'):
        cal_pred = cal_model.predict([[cal_fat,total_fat,sat_fat,trans_fat,cholesterol,sodium,total_carb,protein]])

        st.success(f'Estimasi Kalori : {cal_pred[0]:.2f}')