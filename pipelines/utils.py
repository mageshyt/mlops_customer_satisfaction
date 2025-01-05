import logging

import pandas as pd

from model.data_cleaning import DataCleaning, DataPreProcessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv(
            "/Volumes/Project-2/programming/machine_deep_learning/projects/customer_satisfaction/data/olist_customers_dataset.csv"
        )
        df = df.sample(n=100)
        preprocess_strategy = DataPreProcessStrategy(df)
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.clean_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e


if __name__ == "__main__":
    print(get_data_for_test())
