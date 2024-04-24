import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np

from main import ROOT_PATH

st.info("This is a demo application written to show how to our deeplearing model can be used to predict customer satisfaction scores .")
st.title("Explore Data Analysis 📊")

st.sidebar.title("EDA")


st.image("https://pianalytix.com/wp-content/uploads/2020/11/Exploratory-Data-Analysis.jpg")

df=pd.read_csv(ROOT_PATH+"csv/processed_olist_customers_dataset.csv",)   


st.dataframe(df)
    
# Plot 1: Order Status Distribution

st.subheader("1. Order Status Distribution")

order_status=df['order_status'].value_counts().reset_index()


st.bar_chart(order_status.set_index('index'))

st.divider()


#  Plot 2: Payment Type Distribution

st.subheader("2. Payment Type Distribution")

payment_type=df['payment_type'].value_counts().reset_index()

most_used_payment=payment_type['index'][0]

st.write(f"Most used payment type is {most_used_payment}")

# sns.set_palette("pastel")
# plt.pie(payment_type['payment_type'],labels=payment_type['index'],autopct='%1.1f%%')

# plt.title("Payment Type Distribution")
# st.pyplot(plt)
fig=px.pie(payment_type,values='payment_type',names='index')

st.plotly_chart(fig)
st.divider()
# Plot 3: Order Value Distribution
st.subheader("3. Order Value Distribution")

payment_value=df['payment_value']

st.write(f"Average order value is {payment_value.mean()}")
st.write(f"Maximum order value is {payment_value.max()}")


fig=px.histogram(payment_value,nbins=50)

fig.update_layout(
    title="Order Value Distribution",
    xaxis_title="Order Value",
    yaxis_title="Count"
)

st.plotly_chart(fig)


st.divider()


# Plot 4: Product Category Distribution

st.subheader("4. Product Category Distribution")

product_category=df['product_category_name'].value_counts().reset_index()

# take top 5 product categories

product_category=product_category.head(5)
st.write(f"Most sold product category is {product_category['index'][0]}")

st.bar_chart(
    product_category.set_index('index')
             )

st.divider()


# Plot 5: Review Score Distribution

review_score=df['review_score'].value_counts().reset_index()

# round the review score to nearest integer
review_score['index']=review_score['index'].round(0).astype(int)
st.subheader("5. Review Score Distribution")

print(review_score)
st.write(f"Average review score is {df['review_score'].mean()}")

st.write(f"Most common review score is {review_score['index'][0]}")

colums=["rating 1","rating 2","rating 3","rating 4","rating 5"]
# st.area_chart(review_score.set_index('index'))
st.bar_chart(
    review_score.set_index('index')
             )


# Correlation Matrix

st.subheader("6. Correlation Matrix")

temp_df=df[['payment_sequential','payment_installments','payment_value','freight_value','product_name_lenght','product_description_lenght','review_score']]
correlation_matrix=temp_df.corr()

st.image(ROOT_PATH+"corr.png")

st.write("This is a correlation matrix that shows the relationship between different features in the dataset")

st.write("The correlation matrix shows that there is a strong correlation between payment value and freight value")

