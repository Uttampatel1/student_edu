import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import shap

# Load the personal factors dataset
personal_factors_df = pd.read_csv('data/home.csv')
print(personal_factors_df.columns)

# Preprocess the data
def preprocess_data(df):
    X = df.drop('Score', axis=1)
    y = df['Score']
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Uncomment this line for imputation
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Uncomment this line for imputation
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return X, y, preprocessor

X, y, preprocessor = preprocess_data(personal_factors_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and pipelines
linear_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))])
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))])
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', SVR(kernel='linear'))
])

# Fit models
linear_reg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
gb_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_reg_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_gb = gb_pipeline.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)

# Evaluation
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} - MSE: {mse}, R2: {r2}')

evaluate_model(y_test, y_pred_linear, 'Linear Regression')
evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_gb, 'Gradient Boosting')
evaluate_model(y_test, y_pred_svm, 'SVM with Polynomial Features')

# SHAP Explainer
X_test_preprocessed = rf_pipeline.named_steps['preprocessor'].transform(X_test)
explainer = shap.Explainer(rf_pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_test_preprocessed)
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=preprocessor.get_feature_names_out())
