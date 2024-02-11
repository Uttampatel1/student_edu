import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# Load the personal factors dataset
personal_factors_df = pd.read_csv('data/home.csv')
print(personal_factors_df.columns)

# Preprocess the data
def preprocess_data(df):
    X = df.drop('Score', axis=1)  # Assuming 'Label' is the target column for classification
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
logistic_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('model', LogisticRegression(max_iter=1000, solver='lbfgs'))  # Increase max_iter and specify the solver
])

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(random_state=42))])
gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingClassifier(random_state=42))])
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SVC(kernel='linear'))
])

# Fit models
logistic_reg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
gb_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Predictions
y_pred_logistic = logistic_reg_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_gb = gb_pipeline.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)

# Evaluation
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(f'{model_name} - Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{confusion_mat}')

evaluate_model(y_test, y_pred_logistic, 'Logistic Regression')
evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_gb, 'Gradient Boosting')
evaluate_model(y_test, y_pred_svm, 'SVM')
