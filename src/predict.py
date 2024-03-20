import preprocessing_data
import os
import pickle
import pandas as pd

def load_model(file):
    model_file_path = os.path.join(os.path.dirname(__file__), '..', 'models', file)
    with open(model_file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def preprocess_data(df):
    # Preprocess the data here
    df_processed = preprocessing_data.preprocess_data(df)
    return df_processed

def predict_house_price(df, model):

    # Preprocess the data if needed (e.g., encoding categorical variables)
    # Make predictions
    predicted_price = model.predict(df_processed)
    return predicted_price


# Load the trained Random Forest model
rf_house_model = load_model('trained_RandomForestRegressor(random_state=42)_HOUSE.pkl')

# Define the new house data
new_house_data = {
    'id': [123456],  
    'region': ['Flanders'],  
    'province': ['Antwerp'],  
    'district': ['Antwerp'],  
    'locality': ['Antwerp'],  
    'postalcode': [2000],  
    'latitude': [51.2194],  
    'longitude': [4.4025],  
    'type': ['HOUSE'],  
    'bedrooms': [3.0],  
    'surface': [150.0],  
    'surfaceGood': [200.0],  
    'hasGasWaterElectricityConnection': [1.0],  
    'condition': ['GOOD'],  
    'facadeCount': [2.0],  
    'hasKitchenSetup': ['FULLY_EQUIPPED'],  
    'fireplaceExists': [1],  
    'floodZone': ['NON_FLOOD_ZONE'],  
    'isNewRealEstateProject': [0]  
}

# Create DataFrame for the new house
df_new_house = pd.DataFrame(new_house_data)

# Preprocess the data
df_processed = preprocess_data(df_new_house)

# Convert column names to string type
df_new_house.columns = df_new_house.columns.astype(str)

# Predict the price for the new house
predicted_price = predict_house_price(df_new_house, rf_house_model)

print("Predicted price for the new house:", predicted_price)
