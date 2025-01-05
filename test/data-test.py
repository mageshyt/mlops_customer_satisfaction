import logging
import pytest
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from zenml import pipeline,step

@step
def data_test_prep_step():
    """Test the shape of the data after the data cleaning step."""
    try:
        df = ingest_data()
        # df = ingest_data.get_data()
        # data_cleaning = DataCleaning(df)
        X_train, X_test, y_train, y_test = clean_data(df)

        assert X_train.shape == (
            92487,
            12,
        ), "The shape of the training set is not correct."
        assert y_train.shape == (
            92487,
        ), "The shape of labels of training set is not correct."
        assert X_test.shape == (
            23122,
            12,
        ), "The shape of the testing set is not correct."
        assert y_test.shape == (
            23122,
        ), "The shape of labels of testing set is not correct."
        logging.info("Data Shape Assertion test passed.")
    except Exception as e:
        pytest.fail(e)

@step
def check_data_leakage(X_train, X_test):
    """Test if there is any data leakage."""
    try:
        assert (
                len(X_train.index.intersection(X_test.index)) == 0
        ), "There is data leakage."
        logging.info("Data Leakage test passed.")
    except Exception as e:
        pytest.fail(e)

#
# def test_ouput_range_features(df):
#     """Test output range of the target variable between 0 - 5"""
#     try:
#         assert (
#                 df["review_score"].max() <= 5
#         ), "The output range of the target variable is not correct."
#         assert (
#                 df["review_score"].min() >= 0
#         ), "The output range of the target variable is not correct."
#         logging.info("Output Range Assertion test passed.")
#     except Exception as e:
#         pytest.fail(e)
#

@pipeline(enable_cache=False)
def data_test_pipeline():
    """Data test pipeline."""
    # data_test_prep_step()
    # test_ouput_range_features()
    data_test_prep_step()
    check_data_leakage()

if __name__ == '__main__':
    data_test_pipeline()
