import streamlit as st
from PIL import Image

st.set_page_config(page_title="Customer Satification", page_icon=":shark:", layout="wide")


st.markdown("""
# Preemptive Product Sentiment Forecasting: Understanding Customer Feelings Prior to Purchase
""")

st.image('./assets/high_level_overview.png', caption='Training overview', use_column_width=True)    
st.markdown("""
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

## 📜 Problem statement
---
For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. In order to achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.

## 🐍 Python Requirements
---
Let's jump into the Python packages you need. Within the Python environment of your choice, run:

``` bash
git clone https://github.com/mageshyt/mlops_customer_satisfaction
cd customer-satisfaction/
pip install -r requirements.txt

```
Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to launch the ZenML Server and Dashboard locally, but first you must install the optional dependencies for the ZenML server:
```bash
pip install zenml["server"]
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
        zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
    zenml integration install mlflow -y
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml model-deployer register mlflow --flavor=mlflow
    zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set-default
```


## 👍🏻 Solution:
---
To effectively predict customer satisfaction scores for future orders or purchases, a single model training isn't sufficient. We've devised an end-to-end pipeline for continuously predicting and deploying machine learning models. This pipeline, along with a data application, ensures that the latest model is utilized for business consumption. ZenML facilitates building such pipelines easily and powerfully. Special emphasis is placed on the MLflow integration of ZenML, utilizing MLflow tracking and deployment alongside Streamlit for real-world usage.

        

### 🤖 Training Pipeline
---
Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.


### 🚀 Deployment Pipeline
---
We extend the training pipeline with the deployment pipeline, implementing a continuous deployment workflow. It includes the following additional steps:

- `deployment_trigger`: Checks if the newly trained model meets deployment criteria.
- `model_deployer`: Deploys the model as a service using MLflow if deployment criteria are met.

In this deployment pipeline, ZenML's MLflow tracking integration logs hyperparameter values, the trained model, and model evaluation metrics into the local MLflow backend. It also launches a local MLflow deployment server to serve the latest model if its accuracy is above a configured threshold. The MLflow deployment server runs as a daemon process, automatically updating to serve the new model when a new pipeline is run and the model passes validation.


## 🏗️ Architecture

---

The web application architecture will consist of the following components:

- A frontend web application built using Streamlit
- A machine learning model for predicting customer satisfaction scores


The frontend will interact with the backend 
server through API calls to request predictions,
model training, and data storage.
The backend server will manage user authentication, data storage, and model training. The machine learning model will be trained and deployed using Docker containers.The CI/CD pipeline will be used to automate the deployment process.

## 📌 Pipeline

---
""")

st.image("./assets/training_and_deployment_pipeline_updated.png")


st.markdown("""
    The MLOps (Machine Learning Operations) pipeline project is designed to create an end-to-end workflow for developing and deploying a web application that performs data preprocessing, model training, model evaluation, and prediction. The pipeline leverages Docker containers for encapsulating code, artifacts, and both the frontend and backend components of the application


## 📚 References
---
- [ZenML documentation](https://docs.zenml.io/)
- [MLflow documentation](https://www.mlflow.org/docs/latest/index.html)
- [Streamlit documentation](https://docs.streamlit.io/en/stable/index.html)
- [Streamlit Steup](https://www.youtube.com/watch?v=xTKoyfCQiiU)
- [ZenML Setup](https://www.youtube.com/watch?v=-dJPoLm_gtE&t=7996s)
""")

