import logging

import pandas as pd

from zenml import step 



class IngestData:
    """
    Ingesting the data from the source.
    """

    def __init__(self,data_path:str) -> None:
        """

        Args:
            data_path (str): path to the data source
        """

        self.data_path = data_path

    def get_data(self):
        """
        Reading the data from the source.
        """
        logging.info(f"Reading data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingesting the data from the source.

    Args:
        data_path (str): path to the data source

    Returns:
        pd.DataFrame: data from the source
    """
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        return None
    


