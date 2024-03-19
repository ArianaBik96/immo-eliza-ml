# IMMOWEB Machine Learning
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Description
This project focuses on creating a machine learning model to predict real estate property prices in Belgium using data from Immoweb.
It uses RandomForestRegression to train the model.
At the moment, the R^2 score of the test set is 0.61

![Alt text](pics/machine_learning_1.png)


## 📦 Repo structure
    .
    ├── data
    │   └── data_properties.csv
    │
    ├── pics
    │   └── machine_learning_1.png
    │       
    ├── src
    │   ├── immo_ml.ipynb
    │   ├── predict.py
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

## 🛎️ Usage
To use this repository, follow these steps:

1. **Clone the Repository**: 
    - Clone the repository to your local machine using the following command:
    ```bash
    git clone https://github.com/mahsanazar/immo-eliza-DAMI-analysis.git
    ```

2. **Navigate to the Project Directory**:
    - Once cloned, navigate to the project directory:
    ```bash
    cd immo-eliza-DAMI-analysis
    ```

3. **Explore Analysis Notebooks**:
    - The `analysis` directory contains Jupyter notebooks (`*.ipynb`) where you can explore various analyses performed on the data. Open these notebooks in Jupyter Notebook or JupyterLab to view the analyses and results.

4. **Access Reports**:
    - The `reports` directory includes reports generated from the analysis. These reports may contain visualizations, insights, and conclusions drawn from the data analysis.

5. **Work with Data**:
    - The `data` directory contains the dataset used for analysis. You can find both raw and clean versions of the dataset. Explore the data files to understand their structure and contents.

## 📑 Sources
- [Immoweb](https://www.immoweb.be/en) - Real estate website from which data is scraped.


## ⏱️ Timeline
This project took 5 days for completion.

## 📌 Personal Situation
This project was done as part of the AI Bootcamp at BeCode.org.