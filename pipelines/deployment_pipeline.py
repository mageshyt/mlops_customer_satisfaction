import os
import json

import numpy as np
import pandas as pd

from zenml  import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output


from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import evaluate_model

from .utils import get_data_for_test


docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data



class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0.9


@step
def deployment_trigger(
    accuracy: float,
    config:DeploymentTriggerConfig
) -> bool:
        """Implements a simple model deployment trigger that looks at the 
        input model accuracy and decides if it is good enough to deploy"""
        return accuracy >= config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
     pipeline_name: str,
        step_name: str,
        running: bool=True,
        model_name: str="model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the mlflow model
    model_deployer=MLFlowModelDeployer.get_active_model_deployer()
    # fetch the mlflow deployment service
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(
     service:MLFlowDeploymentService,
     data:np.ndarray
) -> np.ndarray:
     
    service.start(timeout=10)  # should be a NOP if already started 

    data = json.loads(data) # load the data
    data.pop("columns") # remove the columns
    data.pop("index") # remove the index
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ] # select the columns from the data
    df = pd.DataFrame(data["data"], columns=columns_for_df) # create a dataframe
    json_list = json.loads(json.dumps(list(df.T.to_dict().values()))) # convert the dataframe to a list of dictionaries
    data = np.array(json_list) # convert the list of dictionaries to a numpy array
    prediction = service.predict(data) # predict the data
    return prediction # return the prediction

@pipeline(enable_cache=True,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """Continuous deployment pipeline to train and deploy a model, or to
    only run a prediction against the deployed model."""
    df=ingest_data()
    # Cleaning the data
    X_train,X_test,y_train,y_test= clean_data(df)
    # Training the model
    model=model_train(X_train,X_test,y_train,y_test)

    r2,mae,rmse,mse=evaluate_model(model,X_test,y_test)
    # r2 -> measure teh goodness of fit of the model
    deployment_decision=deployment_trigger(r2)

    # Deploy the model
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
