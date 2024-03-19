
import train

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
'''
RandomForestRegressor is a machine learning algorithm that builds multiple decision trees and combines their predictions to make accurate 
guesses about numerical outcomes. Each tree is trained on a random subset of the data and features, reducing the risk of overfitting. 
By aggregating the predictions of these diverse trees, RandomForestRegressor provides robust and reliable results, making it a popular 
choice for regression tasks in various domains.
'''

df_house, df_apartment = train.importing_data()

print('HOUSE')
print('**' *50)
X_train_house, X_test_house, y_train_house, y_test_house= train.get_clean_training_test_data(df_house)
print('APARTMENT')
print('**' *50)
X_train_apartment, X_test_apartment, y_train_apartment, y_test_apartment = train.get_clean_training_test_data(df_apartment)

# Train a Random Forest Regressor for df_house
model_house = RandomForestRegressor(random_state=42)
model_house.fit(X_train_house, y_train_house)

# Predict on the training set for df_house
y_train_pred_house = model_house.predict(X_train_house)

# Calculate training R-squared score for df_house
train_r2_house = r2_score(y_train_house, y_train_pred_house)
print("Training R-squared score for df_house:", train_r2_house)

# Predict on the testing set for df_house
y_pred_house = model_house.predict(X_test_house)

# Calculate R-squared score for df_house
r2_house = r2_score(y_test_house, y_pred_house)
print("R-squared score for df_house:", r2_house)

# Train a Random Forest Regressor for df_apartment
model_apartment = RandomForestRegressor(random_state=42)
model_apartment.fit(X_train_apartment, y_train_apartment)

# Predict on the training set for df_apartment
y_train_pred_apartment = model_apartment.predict(X_train_apartment)

# Calculate training R-squared score for df_apartment
train_r2_apartment = r2_score(y_train_apartment, y_train_pred_apartment)
print("Training R-squared score for df_apartment:", train_r2_apartment)

# Predict on the testing set for df_apartment
y_pred_apartment = model_apartment.predict(X_test_apartment)

# Calculate R-squared score for df_apartment
r2_apartment = r2_score(y_test_apartment, y_pred_apartment)
print("R-squared score for df_apartment:", r2_apartment)


