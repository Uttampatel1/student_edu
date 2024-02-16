import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Function to add noise to a DataFrame
def add_noise(df, noise_factor=0.1):
    noisy_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        noise = np.random.normal(scale=noise_factor, size=len(df))
        noisy_df[col] = df[col] + noise

    return noisy_df


def apply_pca(df, n_components=None):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

    # Separate numerical and categorical data
    X_numerical = df[numerical_cols]
    X_categorical = df[categorical_cols]

    # Set n_components to the minimum of the number of samples and features if None
    if n_components is None:
        n_components = min(X_numerical.shape[0], X_numerical.shape[1])

    # Apply PCA to numerical data
    pca_numerical = PCA(n_components=n_components)
    transformed_data_numerical = pca_numerical.fit_transform(X_numerical)
    df_pca_numerical = pd.DataFrame(transformed_data_numerical, columns=[f'PCA_{i}' for i in range(1, n_components + 1)])

    # Combine numerical PCA with categorical columns
    df_pca = pd.concat([df_pca_numerical, X_categorical.reset_index(drop=True)], axis=1)

    return df_pca


# Load the original data
original_df = pd.read_csv("../data/final.csv")

# Create low-fit and high-fit datasets with PCA
low_fit_df_pca = add_noise(original_df, noise_factor=0.5)  # Increase noise for low fit
high_fit_df_pca = original_df.copy()  # No noise for high fit

low_fit_df_pca = apply_pca(low_fit_df_pca)
high_fit_df_pca = apply_pca(high_fit_df_pca)

# Save modified datasets with PCA
low_fit_df_pca.to_csv("../data/low_fit_data_pca.csv", index=False)
high_fit_df_pca.to_csv("../data/high_fit_data_pca.csv", index=False)
