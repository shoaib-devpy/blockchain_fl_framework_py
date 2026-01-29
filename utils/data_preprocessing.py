import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import os

def _download_kaggle_dataset():
    # Download latest version
    path = kagglehub.dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    print("Path to dataset files:", path)
    return path

def load_and_preprocess_data(file_path):
    # Load the dataset
    if not file_path or not os.path.exists(file_path):
        dataset_dir = _download_kaggle_dataset()
        # Try to find a CSV inside the downloaded dataset
        csv_candidates = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV files found in downloaded dataset: {dataset_dir}")
        file_path = os.path.join(dataset_dir, csv_candidates[0])
    df = pd.read_csv(file_path)
    
    # Preprocess the data (e.g., normalization, handling missing values)
    df.fillna(0, inplace=True)  # Simple example of handling missing values
    
    # Address class imbalance
    df_majority = df[df["Class"] == 0]
    df_minority = df[df["Class"] == 1]
    
    df_minority_upsampled = df_minority.sample(
        n=len(df_majority),
        replace=True,
        random_state=123
    )

    df_upsampled = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)
    
    return df_upsampled

def partition_data_for_clients(df, n_clients):
    client_data = []
    for i in range(n_clients):
        client_df = df.sample(frac=1/n_clients, replace=False, random_state=i)
        client_data.append(client_df)
        df = df.drop(client_df.index)
    return client_data

def prepare_client_data(client_data):
    client_partitions = []
    for client_df in client_data:
        X = client_df.drop('Class', axis=1)
        y = client_df['Class']
        
        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        client_partitions.append((X_train, X_test, y_train, y_test))
    return client_partitions
