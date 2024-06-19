# utils/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv("C:/Users/shoai/Desktop/DC_WEB/creditcard_2023.csv")
    
    # Data Cleaning
    # Handle missing values by filling with zeros (example)
    df.fillna(0, inplace=True)
    
    # Address class imbalance
    df_majority = df[df.Class == 0]
    df_minority = df[df.Class == 1]
    
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=123) # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Feature Engineering
    # Example of creating time-based features (assuming a 'date' column exists)
    if 'date' in df_upsampled.columns:
        df_upsampled['date'] = pd.to_datetime(df_upsampled['date'])
        df_upsampled['day_of_week'] = df_upsampled['date'].dt.dayofweek
        df_upsampled['month'] = df_upsampled['date'].dt.month
        df_upsampled['quarter'] = df_upsampled['date'].dt.quarter

    # Example of aggregating transaction features (assuming 'transaction_amount' column exists)
    if 'transaction_amount' in df_upsampled.columns:
        df_upsampled['transaction_amount_last_week'] = df_upsampled.groupby('user_id')['transaction_amount'].rolling(window=7).sum().reset_index(level=0, drop=True)

    return df_upsampled

def partition_data_for_clients(df, n_clients):
    # Split data into client-specific datasets
    client_data = []
    df_shuffled = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe
    split_size = len(df_shuffled) // n_clients

    for i in range(n_clients):
        client_df = df_shuffled.iloc[i * split_size: (i + 1) * split_size]
        client_data.append(client_df)
    
    return client_data

def prepare_client_data(client_data):
    client_partitions = []

    for data in client_data:
        X = data.drop(['Class', 'date'], axis=1, errors='ignore')
        y = data['Class']
        
        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        client_partitions.append((X_train, X_test, y_train, y_test))

    return client_partitions
