import streamlit as st
import pandas as pd
import plotly.express as px
from annotated_text import annotated_text


from main import ROOT_PATH

st.set_page_config(page_title="Performance Measures", layout="wide")
st.title('Performance Measures 🚧')
st.image('https://miro.medium.com/v2/resize:fit:1400/1*g9x71-cqZ5im3EaFxwyf9Q.jpeg')

# load the performance metrics df



performance_df = pd.read_csv(f"{ROOT_PATH}/csv/performance_metrics.csv")


# 1. overall performance of the model

# Model,MAE,MSE,RMSE,R2,RMSLE,MAPE,TT (Sec)
st.subheader(
    "1. Overall performance of the model (MAE, MSE, RMSE, R2, RMSLE, MAPE, TT (Sec))")




# display the performance metrics
st.dataframe(performance_df)

# best model

annotated_text(("Best Model:" ,"AdaBoost Regressor,Linear Regression","purple"))

# plot the performance  metrics - time 

st.write("1. Performance metrics - Time taken for each model")

st.bar_chart(performance_df,
             x= 'Model',
                y='TT (Sec)',
                color="#ffaa00"
             )
# st.markdown("**Note:** The above metrics are calculated for the test dataset.")
annotated_text(
    ("Note:","The above metrics are calculated for the test dataset.") ,
    
)
# 2. Performance metrics - MAE and MSE

st.write("2. Performance metrics - MAE and MSE")

fig=px.bar_polar(performance_df,
                    r='MAE',
                    theta='Model',
                    color='Model',
                    template="plotly_dark",
                    title="Performance metrics - MAE"
                    )


st.plotly_chart(fig)

st.markdown(
    """**Note:** metric used to measure the difference between actual and projected values in a dataset. 
            """)

annotated_text(
    
    ("Lower values indicate better performance.","")
)

 

st.bar_chart(performance_df,
             x= 'Model',
                y='MSE',

             )
st.markdown(
    """**Note:** The mean squared error (MSE) is a measure of how close a fitted line is to data points. 
            Lower values indicate better performance.
            """)




st.divider()


# 3. Performance metrics - RMSE and R2

st.write("3. Performance metrics - RMSE and R2")

# hist plot

st.bar_chart(performance_df,x="Model",y="RMSE")



st.markdown(
    """**Note:** The root-mean-square error (RMSE) is a measure of how well a model fits the data. 
            """)

annotated_text(
    ("Lower values indicate better performance.","")
)

fig= px.bar_polar(performance_df,
                    r='R2',
                    theta='Model',
                    color='Model',
                    template="plotly_dark",
                    title="Performance metrics - R2"
                    )

st.plotly_chart(fig)

st.markdown(
    """
    **Note:**  R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
    """
)

annotated_text(
    ("Higher values indicate better performance.","")
)


# Provide explanations for the performance of each model