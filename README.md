# IMMOWEB Machine Learning
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Description
This project focuses on creating a machine learning model to predict real estate property prices in Belgium using data from Immoweb.
It uses Linear Regression and RandomForestRegression to train the model.
At the moment, best the R^2 score of the test set for houses is 0.74 and for apartments is 0.71

![Alt text](pics/machine_learning_1.png)


## 📦 Repo structure
    .
    ├── data
    │   └── data_properties.csv
    │
    ├── models
    │   ├── trained_LinearRegression()_APARTMENT.pkl.gz
    │   ├── trained_LinearRegression()_HOUSE.pkl.gz
    │   ├── trained_RandomForestRegressor(random_state=42)_APARTMENT.pkl.gz
    │   └── trained_RandomForestRegressor(random_state=42)_HOUSE.pkl.gz
    │
    ├── pics
    │   └── machine_learning_1.png
    │       
    ├── src
    │   ├── immo_ml.ipynb
    │   ├── predict.py
    │   ├── preprocessing_data.py
    │   └── train.py
    │
    ├── .gitignore
    ├── README.md

## ⚙️ Installation
To get started with Immo Eliza ML, follow these simple steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/ArianaBik96/immo-eliza-ml.git
    ```

2. Navigate to the project directory:
    ```bash
    cd immo-eliza-ml
    ```

3. Install Dependencies:
    Make sure you have Python 3.x installed on your system.
    Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Exploration and Usage:
    You're all set! You can now train your model using train.py, make predictions using predict.py, and work with the data in the data directory. Enjoy!



## ⏱️ Timeline
This project took 5 days for completion.

## 📌 Personal Situation
This project was done as part of the AI Bootcamp at BeCode.org.