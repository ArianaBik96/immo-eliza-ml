import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# Function to impute missing values
def impute_missing_values(df):
    # Impute missing values for numerical features
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])

    # Impute missing values for categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_features = df.select_dtypes(include=['object']).columns
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

    return df

# Function to encode categorical features
def encode_data(df):
    # Collect categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Initialize OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)

    # Fit OneHotEncoder on the data and transform it
    X_encoded = onehot_encoder.fit_transform(df[categorical_columns])

    # Convert the encoded array back into a DataFrame
    X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))

    # Concatenate the encoded categorical variables with the original numerical variables
    X_encoded_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_encoded_full = pd.concat([df.drop(columns=categorical_columns), X_encoded_df], axis=1)

    return df_encoded_full


# Function to preprocess the data
def preprocess_data(df):

    df = impute_missing_values(df)
    print(f' impute: {df.head()}')
    df = encode_data(df)
    print(f' encode: {df.head()}')
    return df