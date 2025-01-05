import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import uuid
def load_and_preprocess_data(filepath):
    print("📥 Loading the dataset...")
    df = pd.read_csv(filepath)
    
    print("🧹 Handling missing values...")
    df['product_weight_g'].fillna(df['product_weight_g'].mean(), inplace=True)
    df['product_length_cm'].fillna(df['product_length_cm'].mean(), inplace=True)
    df['product_height_cm'].fillna(df['product_height_cm'].mean(), inplace=True)
    df['product_width_cm'].fillna(df['product_width_cm'].mean(), inplace=True)
    df.drop(columns=['review_comment_message'], inplace=True)
    df=df.select_dtypes(exclude=['object'])

    
    print("🔄 Encoding categorical variables...")
    categorical_features = df.select_dtypes(include=['object']).columns
    numerical_features = df.select_dtypes(include=['number']).columns
    numerical_features = numerical_features.drop('review_score')  # Exclude the target column
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return df, preprocessor

def train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor):
    print("🚀 Training the model with the best parameters...")
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    print("🔍 Making predictions...")
    train_predictions = model_pipeline.predict(X_train)
    test_predictions = model_pipeline.predict(X_test).round().astype(int)
    
    print("📊 Calculating RMSE and R2 score...")
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    return model_pipeline, train_rmse, test_rmse, train_r2, test_r2

def save_model(model, filepath):
    print("💾 Saving the model...")
    with open(filepath, 'wb') as file:
        joblib.dump(model, file)

def load_model(filepath):
    print("📂 Loading the model...")
    with open(filepath, 'rb') as file:
        return joblib.load(file)
    

if __name__ == "__main__":
    filepath = '/Volumes/Project-2/programming/machine_deep_learning/projects/customer_satisfaction/data/olist_customers_dataset.csv'

    df, preprocessor = load_and_preprocess_data(filepath)
    X = df.drop(columns=['review_score'])
    y = df['review_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model, train_rmse, test_rmse, train_r2, test_r2 = train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor)
    
    save_model(model, f"finalized_model-{uuid.uuid4()}.pkl")
    print("Training RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Training R2:", train_r2)
    print("Test R2:", test_r2)