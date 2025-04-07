# Machine Learning Models for Regression Analysis

This project involves training and evaluating several regression models to predict the target variable `Cmoy` from a dataset of small machines. The dataset consists of different machine features, and the goal is to predict the average machine output (`Cmoy`) using various regression models, such as Random Forest, Gradient Boosting, XGBoost, and more. Additionally, the impact of feature engineering (FE) on model performance is also evaluated.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [Code Explanation](#code-explanation)

## Prerequisites

To run this project, you need to install the required dependencies. You can install them via the `requirements.txt` file.

### `requirements.txt`
```plaintext```
pandas
scikit-learn
tqdm
matplotlib
eli5
IPython
## Dataset

The dataset used for this project is a collection of small machines. The target variable is `Cmoy`, and the dataset contains various numerical features related to the machines. The dataset is split into a training set and a test set.

- **Training Set**: `Dataset_numerique_20000_petites_machines.csv`
- **Test Set**: `Dataset_numerique_10000_petites_machines.csv`

The target column `Cmoy` is separated from the features in both datasets. The datasets are read using the pandas library and processed to separate the target variable and features.

## Models Used

The following regression models are tested to predict the target variable `Cmoy`:

- **Ridge Regression**
- **Random Forest Regression**
- **Gradient Boosting Regression**
- **HistGradient Boosting Regression**
- **Extra Trees Regression**
- **XGBoost**
- **LightGBM**
- **Lasso Regression**

Each model is trained and evaluated using both the original features and features after engineering (FE) to analyze the impact of feature modifications on model performance.

## Feature Engineering

Feature engineering is applied to improve model performance. This step involves transforming the input data to help the models understand the underlying patterns more effectively. In this case, the feature engineering process is implemented before training the models, and its effect on performance is compared against the baseline (without feature engineering).

## Results

The performance of the models is evaluated based on their R² score, both with and without feature engineering. The following table summarizes the performance comparison:

| Model                | Score Without FE | Score With FE | Improvement         |
|----------------------|------------------|---------------|---------------------|
| Ridge                | 0.94057          | 0.94508       | 0.00451             |
| Random Forest        | 0.92799          | 0.98031       | 0.05231             |
| Gradient Boosting    | 0.94515          | 0.98160       | 0.03644             |
| HistGradient Boosting| 0.95864          | 0.98423       | 0.02559             |
| Extra Trees          | 0.80914          | 0.97991       | 0.17078             |
| XGBoost              | 0.95280          | 0.98258       | 0.02978             |
| LightGBM             | 0.95596          | 0.98350       | 0.02754             |
| Lasso                | 0.94049          | 0.94509       | 0.00460             |

From the results, it is clear that feature engineering improves model performance significantly, especially for models like Random Forest and Extra Trees.

## Code Explanation

The code is designed to:

1. Load the dataset and preprocess it (splitting into features and target).
2. Train the models with the original features and then with engineered features.
3. Evaluate the models based on R² score to determine the effectiveness of feature engineering.
4. Display the results in a table comparing model performance before and after feature engineering.

Each part of the code is modular and allows for easy experimentation with different models and feature engineering techniques. The key steps include data preparation, model training, and performance evaluation, ensuring a systematic approach to model comparison.
