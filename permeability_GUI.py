import streamlit as st
import pickle, requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor

url = "https://raw.githubusercontent.com/cakirogl/SCM/refs/heads/main/SCM_Cl28_only_Coulomb.csv"
model_selector = st.selectbox('**Predictive model**', ["XGBoost", "CatBoost"])
quantile_selector = st.selectbox('**Select quantile**', ["80%", "90%", "95%"])
df = pd.read_csv(url);
x, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler = MinMaxScaler();
x=scaler.fit_transform(x);
input_container = st.container()
ic1,ic2,ic3 = input_container.columns(3)
with ic1:
    cement = st.number_input("**Cement [kg/m$^3$]:**",min_value=75.0,max_value=715.0,step=5.0,value=315.77)
    water = st.number_input("**Water [kg/m$^3$]:**",min_value=60.0,max_value=200.0,step = 5.0,value=171.08)
    c_agg = st.number_input("**Coarse agg. [kg/m$^3$]:**", min_value=500.0, max_value=1400.0, step=20.0, value=830.87)
    f_agg = st.number_input("**Fine agg. [kg/m$^3$]:**", min_value=400.0, max_value=1200.0, step=20.0, value=846.81)
    FA = st.number_input("**Fly ash [kg/m$^3$]:**", min_value=0.0, max_value=450.0, step=20.0, value=96.19)
with ic2:
    SF = st.number_input("**Silica fume [kg/m$^3$]:**", min_value=0.0, max_value=150.0, step=20.0, value=23.04)
    GGBFS=st.number_input("**Ground granulated blast furnace slag [kg/m$^3$]:**", min_value=0.0, max_value=400.0, step=20.0, value=59.62)
    SP = st.number_input("**Superplasticizer [kg/m$^3$]**", min_value=0.0, max_value=12.0, step=1.0, value=4.67)

new_sample=np.array([[cement, water, c_agg, f_agg, FA, SF, GGBFS, SP]],dtype=object)
new_sample=pd.DataFrame(new_sample, columns=df.columns[:-1])
new_sample=scaler.transform(new_sample);
if model_selector=="XGBoost":
    #model=XGBRegressor(random_state=0)
    #model.fit(x, y)
    q80 = 369.638; q90 = 587.642;  q95 = 928.972;
    xgb_url = "https://raw.githubusercontent.com/cakirogl/SCM/refs/heads/main/XGBmodel.pkl"
    response = requests.get(xgb_url)
    model = pickle.loads(response.content)
if model_selector=="CatBoost":
    q80 = 249.937; q90 = 405.552;  q95 = 1038.826;
    #model=CatBoostRegressor(random_state=0, verbose=0)
    #model.fit(x,y)
    cb_url = "https://raw.githubusercontent.com/cakirogl/SCM/refs/heads/main/CBmodel.pkl"
    response = requests.get(cb_url)
    model = pickle.loads(response.content)


with ic2:
    #st.write(f":blue[**Compressive strength = **{model_c.predict(new_sample)[0]:.3f}** MPa**]\n")
    point_pred = model.predict(new_sample)[0];
    if quantile_selector == "80%":
        st.write(f":blue[**Chloride permeability interval = [**{point_pred-q80:.0f}**C, **{point_pred+q80:.0f}**C]**]\n")
    if quantile_selector == "90%":
        st.write(f":blue[**Chloride permeability interval = [**{point_pred-q90:.0f}**C, **{point_pred+q90:.0f}**C]**]\n")
    if quantile_selector == "95%":
        st.write(f":blue[**Chloride permeability interval = [**{point_pred-q95:.0f}**C, **{point_pred+q95:.0f}**C]**]\n")
    
#with ic3:
    #st.write(f":blue[**Tensile strength = **{model_t.predict(new_sample)[0]:.3f}** MPa**]\n")
#    st.write(f":blue[**Tensile strength = **{model_t.predict(new_sample_t)[0]:.3f}** MPa**]\n")