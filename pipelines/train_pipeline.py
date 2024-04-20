from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import evaluate_model


# Create a pipeline

@pipeline(enable_cache=True)
def train_pipeline(data_path:str) -> None:
    """
    Training pipeline.
    
    Args:
        data_path (str): path to the data source
    """
    df=ingest_data(data_path) # Ingesting the data
    # Cleaning the data
    X_train,X_test,y_train,y_test= clean_data(df)
    # Training the model
    model=model_train(X_train,X_test,y_train,y_test)

    r2,mae,rmse,mse=evaluate_model(model,X_test,y_test)




