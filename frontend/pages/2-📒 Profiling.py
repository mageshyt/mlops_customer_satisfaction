
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from main import ROOT_PATH
import pandas as pd


st.title("Automated Data Profiling")

df=pd.read_csv(ROOT_PATH+"csv/processed_olist_customers_dataset.csv",)

st.dataframe(df)

pr = ProfileReport(df)

st_profile_report(pr,)










