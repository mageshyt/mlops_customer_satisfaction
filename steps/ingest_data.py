import logging

import pandas as pd

from zenml import step 



class IngestData:
    """
    Data ingestion class which ingests data from
    the souce and returns a DataFrame
    """

    def __init__(self) -> None:
        """
        Initializing the data source.
        """
        pass

    def get_data(self):
        """
        Reading the data from the source.
        """
        logging.info(f"Reading the data ")
        return pd.read_csv("/Volumes/Project-2/programming/machine_deep_learning/projects/customer_satisfaction/data/olist_customers_dataset.csv")
    
@step
def ingest_data() -> pd.DataFrame:
    """
    Ingesting the data from the source.

    Args:
        data_path (str): path to the data source

    Returns:
        pd.DataFrame: data from the source
    """
    try:
        ingest_data = IngestData()
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        return None
    


