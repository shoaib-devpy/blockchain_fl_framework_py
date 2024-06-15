# utils/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv('C:/Users/shoai/Desktop/DC_WEB/creditcard_2023.csv')
    
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
    
    # Split the data into features and labels
    X = df_upsampled.drop('Class', axis=1)
    y = df_upsampled['Class']
    
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
