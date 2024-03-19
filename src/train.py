import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

def get_clean_training_test_data(df):
    create_features(df)
    print(f'Feature {df.head()}')
    print(df.shape)

    # Cleaning data
    df = cleaning_data(df)
    print(f'cleaning {df.head()}')
    print(df.shape)

    X_train, X_test, y_train, y_test = split_into_test_and_training(df)
    print(f'splitting {X_train.head()}')
    print(X_train.shape)

    X_train, X_test = impute_missing_values(X_train, X_test)
    print(f'impute {X_train.head()}')
    print(X_train.shape)

    X_train, X_test = encode_data(X_train, X_test)
    print(f'encode {X_train.head()}')
    print(X_train.shape)

    X_train, X_test = standardize_data(X_train, X_test)
    print(f'standardize {X_train.head()}')
    print(X_train.shape)

    return X_train, X_test, y_train, y_test


def importing_data():
    # Get the path to the data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Construct the full path to data_properties.csv
    file_path = os.path.join(data_dir, 'data_properties.csv')
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter houses and apartments
    df_house = df[df['type'] == 'HOUSE'].copy()
    df_apartment = df[df['type'] == 'APARTMENT'].copy()

    return df_house, df_apartment


def create_features(df):
    df['price_per_sqm'] = df['price_main'] / df['surface']
    
    region_avg_price = df.groupby('region')['price_per_sqm'].transform('mean')
    province_avg_price = df.groupby('province')['price_per_sqm'].transform('mean')
    df['region_avg_price'] = region_avg_price
    df['province_avg_price'] = province_avg_price
    
    # Size-related Features
    df['surface_to_surfaceGood_ratio'] = df['surface'] / df['surfaceGood']
        
    # Property Type-specific Features
    df['terrace_per_sqm'] = df['terraceSurface'] / df['surface']
    
    # Income-related Features
    df['cadastral_income_per_sqm'] = df['cadastralIncome'] / df['surface']
    
    # Energy-related Features
    df['epc_score_impact'] = df['epcScores'].astype('category').cat.codes * df['bedrooms']
    
    return df


def cleaning_data(df):

    # Count NaN values before replacing
    nan_before = df.isna().sum().sum()
    print("Number of NaN values before replacing:", nan_before)

    # Replace infinity or very large values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Count NaN values after replacing
    nan_after = df.isna().sum().sum()
    print("Number of NaN values after replacing:", nan_after)

    # Print the difference
    nan_changed = nan_after - nan_before
    print("Number of values changed to NaN:", nan_changed)
    
    ## Handling Outliers
    Q1 = df['price_main'].quantile(0.25) # Calculate Q1
    Q3 = df['price_main'].quantile(0.75) # Calculate Q3

    IQR = Q3 - Q1 # Calculate IQ range

    lower_bound = Q1 - 1.5 * IQR # Calculate the lower bound
    upper_bound = Q3 + 1.5 * IQR # Calculate the upper bound

    df = df[(df['price_main'] >= lower_bound) & (df['price_main'] <= upper_bound)] # Filter rows where 'price_main' < lb or > ub

    missing_data_all = df.isna().sum()
    percentage_missing_all = round(missing_data_all * 100 / len(df), 0)

    # Drop columns with more than 50% missing values
    df.drop(columns=get_column_missing_values(percentage_missing_all), inplace=True)  # Modify DataFrame inplace

    return df


def get_column_missing_values(percentage_missing_all):
    columns_to_drop = []
    for column, missing_percentage in percentage_missing_all.items():
        if missing_percentage >= 50:
            columns_to_drop.append(column)
    return columns_to_drop


def impute_missing_values(X_train, X_test):

    # Impute missing values for numerical features
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(X_train[numerical_features].head(20))
    X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = numerical_imputer.transform(X_test[numerical_features])

    # Impute missing values for categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_features = X_train.select_dtypes(include=['object']).columns
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])
    return X_train, X_test

def encode_data(X_train, X_test):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Iterate over each column in df_train
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = label_encoder.fit_transform(X_train[col])
            X_test[col] = X_test[col].map(lambda s: label_encoder.transform([s])[0] if s in label_encoder.classes_ else -1)

    return X_train, X_test

def standardize_data(X_train, X_test):
    # Assuming you have a list of column names (feature names) for X_train and X_test
    column_names = X_train.columns.tolist()

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform both training and test sets
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the scaled arrays back into DataFrames
    X_train = pd.DataFrame(X_train, columns=column_names)
    X_test_ = pd.DataFrame(X_test, columns=column_names)

    return X_train, X_test


def split_into_test_and_training(df):
    # Split houses data into training and testing sets
    X_train, X_test = train_test_split(df.drop('price_main', axis=1), test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(df['price_main'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

