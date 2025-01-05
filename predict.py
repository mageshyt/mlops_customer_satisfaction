import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main
# from _utils.utils import get_random_sample





def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    # high_level_image = Image.open("_assets/high_level_overview.png")
    # st.image(high_level_image, caption="High Level Pipeline")

    # whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    # st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |


    """
    )


    payment_sequential = st.sidebar.slider("Payment Sequential", value=0,min_value=0, max_value=29, step=1)
    payment_installments = st.sidebar.slider("Payment Installments", value=1, min_value=0, max_value=24, step=1)
    payment_value = st.number_input("Payment Value", value=209.06)
    price = st.number_input("Price", value=183.29)
    freight_value = st.number_input("freight_value", value=25.77)
    product_name_length = st.number_input("Product name length", value=55.0)
    product_description_length = st.number_input("Product Description length", value=506.0)
    product_photos_qty = st.number_input("Product photos Quantity ", value=1.0)
    product_weight_g = st.number_input("Product weight measured in grams", value=1225.0)
    product_length_cm = st.number_input("Product length (CMs)", value=27.0)
    product_height_cm = st.number_input("Product height (CMs)", value=35.0)
    product_width_cm = st.number_input("Product width (CMs)", value=15.0)
        
    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            # run_main()

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is : {}".format(
                round(pred[0],2)
            )
        )

        
    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.read_csv("/Users/hmagesh/Desktop/programming/Machine-Learning-2023/projects/mlops_customer_satisfaction/frontend/assets/csv/metrics.csv")
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )


if __name__ == "__main__":
    main()