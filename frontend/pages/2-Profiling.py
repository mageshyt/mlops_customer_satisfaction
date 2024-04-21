
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import pandas as pd


st.title("Automated Data Profiling")
ROOT_PATH = "/Volumes/Project-2/programming/machine_deep_learning/projects/customer_satisfaction/frontend/assets/"

df=pd.read_csv(ROOT_PATH+"csv/processed_olist_customers_dataset.csv",)

st.dataframe(df)

pr = ProfileReport(df)

st_profile_report(pr,)










