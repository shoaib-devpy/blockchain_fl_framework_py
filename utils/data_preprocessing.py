# utils/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv("C:/Users/shoai/Desktop/DC_WEB/creditcard_2023.csv")
    
    # Preprocess the data (e.g., normalization, handling missing values)
    df.fillna(0, inplace=True)  # Simple example of handling missing values
    
    # Address class imbalance
    df_majority = df[df.Class == 0]
    df_minority = df[df.Class == 1]
    
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=len(df_majority),    # to match majority class
                                     random_state=123) # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
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
