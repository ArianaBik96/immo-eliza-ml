import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import pickle
import gzip
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def importing_data(file):

    # Get the path to the data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Construct the full path to data_properties.csv

    file_path = os.path.join(data_dir, file)

    file_path = os.path.join(data_dir, 'data_properties.csv')

    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter houses and apartments

    df_house = df[df['type'] == 'HOUSE']
    df_apartment = df[df['type'] == 'APARTMENT']

    return df_house, df_apartment

def get_clean_training_test_data(df, type):
    create_features(df)
    df = cleaning_data(df)
    X_train, X_test, y_train, y_test = split_into_test_and_training(df)
    X_train, X_test, num_imp, cat_imp = impute_missing_values(X_train, X_test,type)
    X_train, X_test, ohe = encode_data(X_train, X_test)
    X_train, X_test, scaler = standardize_data(X_train, X_test)
    
    return X_train, X_test, y_train, y_test, num_imp, cat_imp, ohe, scaler


def create_features(df):
    # Property Type-specific Features
    if 'terraceSurface' in df.columns and 'surface' in df.columns:
        df['terrace_per_sqm'] = df['terraceSurface'] / df['surface']
    
    # Income-related Features
    if 'cadastralIncome' in df.columns and 'surface' in df.columns:
        df['cadastral_income_per_sqm'] = df['cadastralIncome'] / df['surface']
    
    # Energy-related Features
    if 'epcScores' in df.columns and 'bedrooms' in df.columns:
        df['epc_score_impact'] = df['epcScores'].astype('category').cat.codes * df['bedrooms']
       
    return df


def cleaning_data(df):

    ## Handling Outliers
    Q1 = df['price_main'].quantile(0.25) # Calculate Q1
    Q3 = df['price_main'].quantile(0.75) # Calculate Q3

    IQR = Q3 - Q1 # Calculate IQ range

    lower_bound = Q1 - 1.5 * IQR # Calculate the lower bound
    upper_bound = Q3 + 1.5 * IQR # Calculate the upper bound

    df = df[(df['price_main'] >= lower_bound) & (df['price_main'] <= upper_bound)] # Filter rows where 'price_main' < lb or > ub

    missing_data_all = df.isna().sum()
    percentage_missing_all = round(missing_data_all * 100 / len(df), 0)
    df = df.drop(columns=get_column_missing_values(percentage_missing_all)).copy()
    df = df.drop(columns=['id', 'locality', 'postalcode']).copy()

    # List of subtype values to drop
    subtypes_to_drop_houses = ['APARTMENT_BLOCK', 'EXCEPTIONAL_PROPERTY', 'FARMHOUSE', 'COUNTRY_COTTAGE', 'OTHER_PROPERTY', 'KOT', 'SERVICE_FLAT' ]
    # Create a boolean mask to identify rows with the specified subtypes
    mask = df['subtype'].isin(subtypes_to_drop_houses)
    # Drop rows with specified subtypes
    df = df[~mask]

    print(df['subtype'].value_counts())
    
 
    # print(f'COLUMNS IN DF AFTER CLEANING : {list(df.columns)} ')

    return df


def get_column_missing_values(percentage_missing_all):
    columns_to_drop = []
    for column, missing_percentage in percentage_missing_all.items():
        if missing_percentage >= 50:
            columns_to_drop.append(column)
    return columns_to_drop

def get_column_names(X_train):
    return list(X_train.columns)


def impute_missing_values(X_train, X_test,type):

    save_column_names(X_train,type)

    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    print("Numerical Features:", numerical_features)

    # Identify categorical features
    categorical_features = X_train.select_dtypes(include=['object']).columns
    print("Categorical Features:", categorical_features)

    # Impute missing values for numerical features
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

    X_train[numerical_features] = numerical_imputer.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = numerical_imputer.transform(X_test[numerical_features])

    # Impute missing values for categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_features = X_train.select_dtypes(include=['object']).columns
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])



    return X_train, X_test, numerical_imputer, categorical_imputer

def encode_data(X_train, X_test):

    # Collect categorical columns
    categorical_columns = X_train.select_dtypes(include=['object']).columns

    # Initialize OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit OneHotEncoder on the training data and transform both training and testing data
    X_train_encoded = onehot_encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = onehot_encoder.transform(X_test[categorical_columns])

    # Convert the encoded arrays back into DataFrames
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))

    # Concatenate the encoded categorical variables with the original numerical variables
    X_train_encoded_df.reset_index(drop=True, inplace=True)
    X_test_encoded_df.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_train_encoded_full = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded_df], axis=1)
    X_test_encoded_full = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded_df], axis=1)



    return X_train_encoded_full, X_test_encoded_full, onehot_encoder

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

    return X_train, X_test, scaler


def split_into_test_and_training(df):
    # Split houses data into training and testing sets
    X_train, X_test = train_test_split(df.drop('price_main', axis=1), test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(df['price_main'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def linear_regression_model(X_train, X_test, y_train, y_test, type_prop):

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print( "**" * 50)
    print( "LINEAR REGRESSION")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    # Calculate training R-squared score for df_house
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Training R-squared score for {type_prop}:", train_r2)

    # Predict on the testing set for df_house
    y_pred= lr.predict(X_test)

    # Calculate R-squared score for df_house
    r2_house = r2_score(y_test, y_pred)
    print(f"R-squared score for {type_prop}:", r2_house)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE) for {type_prop}:", mse)

    print()
    print('AFTER CROSS VALIDATION')

    # Perform cross-validation
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validated R-squared scores for {type_prop}:", cv_scores)
    print(f"Mean cross-validated R-squared score for {type_prop}:", np.mean(cv_scores))

    # Predict on the testing set
    y_pred = lr.predict(X_test)

    # Calculate R-squared score for test set
    r2_test = r2_score(y_test, y_pred)
    print(f"R-squared score for {type_prop} on test set:", r2_test)

    return lr, r2_test


def random_forest_regression(X_train, X_test, y_train, y_test, type_prop):

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print("**" * 50)
    print("RANDOM FOREST REGRESSION")

    # Train a Random Forest Regressor for df_house
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    # Predict on the training set for df_house
    y_pred_train = rf.predict(X_train)

    # Calculate training R-squared score for df_house
    train_r2 = r2_score(y_train, y_pred_train)
    print(f"Training R-squared score for {type_prop}", train_r2)

    # Predict on the testing set for df_house
    y_pred_test = rf.predict(X_test)

    # Calculate R-squared score for df_house
    r2_test = r2_score(y_test, y_pred_test)
    print(f"R-squared score for {type_prop}", r2_test)

    # Calculate Mean Squared Error (MSE) for the test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Mean Squared Error (MSE) for {type_prop}", mse_test)

    return rf, r2_test


def save_model(model, type_prop):
    model_type = type(model).__name__
    if model_type == 'LinearRegression':
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'linear_regression', f'trained_{model_type}_{type_prop}.pkl.gz')
    else:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_regression', f'trained_{model_type}_{type_prop}.pkl.gz')

    model_bytes = pickle.dumps(model)
    with gzip.open(model_path, 'wb') as f:
        f.write(model_bytes)
    print(f"Model saved successfully at: {model_path}")

def save_preprocessing(num_imp, cat_imp, encoder, scaler, type_prop):
    # Define the file paths for saving the models
    if type_prop == 'HOUSE':
        numImp_file_path = os.path.join('preprocess_models/house', f'trained_num_imp_{type_prop}.pkl.gz')
        catImp_file_path = os.path.join('preprocess_models/house', f'trained_cat_imp_{type_prop}.pkl.gz')
        encoder_file_path = os.path.join('preprocess_models/house', f'trained_encoder_{type_prop}.pkl.gz')
        scaler_file_path = os.path.join('preprocess_models/house', f'trained_scaler_{type_prop}.pkl.gz')
    else:
        numImp_file_path = os.path.join('preprocess_models/apartment', f'trained_num_imp_{type_prop}.pkl.gz')
        catImp_file_path = os.path.join('preprocess_models/apartment', f'trained_cat_imp_{type_prop}.pkl.gz')
        encoder_file_path = os.path.join('preprocess_models/apartment', f'trained_encoder_{type_prop}.pkl.gz')
        scaler_file_path = os.path.join('preprocess_models/apartment', f'trained_scaler_{type_prop}.pkl.gz')

    # Pickle and compress the imputer object
    with gzip.open(numImp_file_path, 'wb') as f:
        pickle.dump(num_imp, f)
    print(f"Imputer saved successfully at: {numImp_file_path}")

        # Pickle and compress the imputer object
    with gzip.open(catImp_file_path, 'wb') as f:
        pickle.dump(cat_imp, f)
    print(f"Imputer saved successfully at: {catImp_file_path}")

    # Pickle and compress the encoder object
    with gzip.open(encoder_file_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Encoder saved successfully at: {encoder_file_path}")

    # Pickle and compress the scaler object
    with gzip.open(scaler_file_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved successfully at: {scaler_file_path}")


def save_column_names(X_train, type):
    print(f'THE TYPE OF THIS IS : {type} ')
    column_names_path = os.path.join(os.path.dirname(__file__), '..', 'column_names', f'column_names_{type}.csv')
    column_names = get_column_names(X_train)
    # Open the file in write mode and write the data
    with open(column_names_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
    
df_house, df_apartment = importing_data('data_properties.csv')

X_train_house, X_test_house, y_train_house, y_test_house, num_imp_house, cat_imp_house, ohe_house, scaler_house= get_clean_training_test_data(df_house, 'HOUSE')

print(list(X_train_house.columns))

model_lr_house, r2_score_lr_house = linear_regression_model(X_train_house, X_test_house, y_train_house, y_test_house, 'HOUSE')
model_rf_house, r2_score_rf_house = random_forest_regression(X_train_house, X_test_house, y_train_house, y_test_house, 'HOUSE')
save_model(model_lr_house, 'HOUSE')
save_model(model_rf_house, 'HOUSE')
save_preprocessing(num_imp_house, cat_imp_house, ohe_house, scaler_house,'HOUSE')

X_train_apartment, X_test_apartment, y_train_apartment, y_test_apartment, num_imp_apartment, cat_imp_apartment, ohe_apartment, scaler_apartment = get_clean_training_test_data(df_apartment,'APARTMENT')
model_lr_ap, r2_score_lr_ap = linear_regression_model(X_train_apartment, X_test_apartment, y_train_apartment, y_test_apartment,'APARTMENT')
model_rf_ap, r2_score_rf_ap = random_forest_regression(X_train_apartment, X_test_apartment, y_train_apartment, y_test_apartment,'APARTMENT')


save_model(model_lr_ap, 'APARTMENT')
save_model(model_rf_ap, 'APARTMENT')
save_preprocessing(num_imp_apartment, cat_imp_apartment, ohe_apartment, scaler_apartment, 'APARTMENT')



