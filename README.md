# Crypto-BuySell-Model
This serves as a capstone project for my Data Science/ML course at LuxDev HQ.

The goal is to create a fully functioning ML system that:

-   Fetches real historical crypto data
-   Cleans and engineers features
-   Calculates technical indicators
-   Generates labels for Buy/Sell/Hold
-   Trains a classification model
-   Evaluates performance and backtests strategy
-   Serializes the trained model
-   Deploys prediction logic (optional)

This is a realistic applied-finance ML workflow similar to what quant
researchers build.

## Project Documentation
To clone this project;
- Copy the URL code from the GitHub repo
- Use the 'git clone' command to clone the repository into your local computer i.e git clone 'paste copied URL'

This project uses the following libraries with all the versions detailed in the requirements.txt file in the main branch:
- OS
- Numpy
- Pandas
- TA-Lib

The project is divided into notebooks as well as src, which are different folders.
The notebooks and src folders and files map as follows:

- 01_fetch_data.ipynb -> data_fetcher.py
- 02_feature_engineering.ipynb -> feature_generator.py
- 03_model_training.ipynb -> labeler.py & predict.py
- 04_evaluation.ipynb -> train.py

Each file contains comments which give a more logical meaning to the code allowing for better understanding of the code.

These stand as unexplained in this project
- df.info() -  provides a concise summary of a DataFrame

- df.head() -  displays the first few rows of a DataFrame

- df.tail() -  displays the last few rows of a DataFrame
