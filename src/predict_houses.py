from collections import OrderedDict
import os
import csv
import gzip
import pickle
import pandas as pd

def load_model(file):
    # Define the file path for the compressed model
    model_file_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_regression', file)
    # Check if the file exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"The specified model file '{model_file_path}' does not exist.")
    # Load the compressed model file
    with gzip.open(model_file_path, 'rb') as f:
        loaded_model_bytes = f.read()
    # Unpickle the model bytes
    loaded_model = pickle.loads(loaded_model_bytes)

    return loaded_model

def load_preprocessing(cat_file_path, num_file_path, enc_file_path, stand_file_path):
    cat_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', 'house', cat_file_path)
    num_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', 'house', num_file_path)
    enc_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', 'house', enc_file_path)
    stand_file_path = os.path.join(os.path.dirname(__file__), '..', 'preprocess_models', 'house', stand_file_path)

    # Load categorical imputer
    with gzip.open(cat_file_path, 'rb') as f:
        loaded_cat_imp = pickle.load(f)

    # Load numerical imputer
    with gzip.open(num_file_path, 'rb') as f:
        loaded_num_imp = pickle.load(f)

    # Load one-hot encoder
    with gzip.open(enc_file_path, 'rb') as f:
        loaded_enc = pickle.load(f)

    # Load scaler
    with gzip.open(stand_file_path, 'rb') as f:
        loaded_stand = pickle.load(f)

    return loaded_cat_imp, loaded_num_imp, loaded_enc, loaded_stand

def load_column_names(column_names_path):
    # Open the CSV file and read its contents into a list
    with open(column_names_path, 'r') as file:
        reader = csv.reader(file)
        # Assuming the CSV file contains a single row of column names
        column_names = next(reader)
    return column_names

def preprocess_data(df, column_names):
    # Load preprocessing objects
    trained_cat_imputer, trained_num_imputer, trained_onehot_encoder, trained_scaler = load_preprocessing('trained_cat_imp_HOUSE.pkl.gz', 'trained_num_imp_HOUSE.pkl.gz', 'trained_encoder_HOUSE.pkl.gz', 'trained_scaler_HOUSE.pkl.gz')

    existing_columns = list(df.keys())
    # Find the missing columns
    missing_columns = [col for col in column_names if col not in existing_columns]

    # Add the missing columns to new_house_data and assign them a value of 0
    for col in missing_columns:
        df[col] = 0

    print(f'THESE ARE THE COLUMNS OF column_names {column_names}')

    # Order the columns of new_house_data as per column_names
    df = OrderedDict((col, df[col]) for col in column_names)

    # Convert to DataFrame
    df = pd.DataFrame(df)

    # For categorical imputer (`trained_cat_imputer`)
    # Assuming `trained_cat_imputer` is your trained SimpleImputer object for categorical data
    categorical_cols = trained_cat_imputer.get_feature_names_out()
    print("Columns of the categorical imputer:", categorical_cols)

    numerical_cols =[col for col in column_names if col not in categorical_cols]
    
    print(f' THESE ARE THE NUMERICAL COLUMNS {numerical_cols}')

    # Apply imputation for numerical features
    df[numerical_cols] = trained_num_imputer.transform(df[numerical_cols])

    # Apply imputation for categorical features
    df[categorical_cols] = trained_cat_imputer.transform(df[categorical_cols])

    # Apply one-hot encoding
    df_encoded = trained_onehot_encoder.transform(df[categorical_cols])
    df_encoded_df = pd.DataFrame(df_encoded, columns=trained_onehot_encoder.get_feature_names_out(categorical_cols))
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, df_encoded_df], axis=1)

    # Apply standardization
    df = trained_scaler.transform(df)

    return df


def predict_house_price(df, model):
    column_names_path = os.path.join(os.path.dirname(__file__), '..', 'column_names', 'column_names_HOUSE.csv')
    column_names = load_column_names(column_names_path)

    # Preprocess the data
    df_processed = preprocess_data(df, column_names)
    
    # Make predictions
    predicted_price = model.predict(df_processed)
    # Convert the predicted price to a scalar value
    predicted_price_scalar = predicted_price[0]
    # Format the predicted price with dot as decimal separator and comma as thousand separator
    formatted_price = "{:,.2f}".format(predicted_price_scalar).replace(",", " ")
    return formatted_price


# Load the trained Random Forest model
rf_house_model = load_model('trained_RandomForestRegressor_HOUSE.pkl.gz')


# Define the new house data
new_house_data = { 
    'district': ['Antwerp'], 
    'epcScores': ['B'],
    'bedrooms': [3],
    'surface': [150],
    'facadeCount': [2],  
}

# Create DataFrame for the new house
df_new_house = pd.DataFrame(new_house_data)

# Predict the price for the new house
predicted_price = predict_house_price(df_new_house, rf_house_model)

print("Predicted price for the new house:", predicted_price)

