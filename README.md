# House Price Prediction

This project aims to predict house prices using a linear regression model. The dataset used is the Boston Housing dataset.

## Overview

This project involves training a linear regression model to predict the median value of owner-occupied homes in Boston suburbs. The steps involved in the project are:

1. Loading the dataset
2. Defining features and the target variable
3. Splitting the dataset into training and testing sets
4. Training a linear regression model
5. Making predictions on the test set
6. Evaluating the model's performance

## Dataset

The dataset used in this project is the Boston Housing dataset. It includes various features of houses in Boston and the target variable is the median value of owner-occupied homes (`medv`).

## Installation

To run the project, you need to have the following libraries installed:

- pandas
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas scikit-learn
```

## Usage

To run the project, execute the Jupyter notebook file `House_Price_Prediction.ipynb`. The notebook includes code cells that perform the following steps:

1. **Loading the dataset:**
    ```python
    import pandas as pd

    data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
    ```

2. **Defining features and target:**
    ```python
    X = data.drop(columns=['medv'])
    y = data['medv']
    ```

3. **Splitting the dataset:**
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. **Training the model:**
    ```python
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

5. **Making predictions:**
    ```python
    y_pred = model.predict(X_test)
    ```

6. **Evaluating the model:**
    ```python
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    ```

## Results

The model's performance is evaluated using Mean Squared Error (MSE). The printed output in the notebook indicates the MSE value.

---
