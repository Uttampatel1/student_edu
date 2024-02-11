import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(random_state=42))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse, "R2:", r2)
    return y_pred, y_test

def categorize_scores(scores):
    categories = []
    for score in scores:
        if score < 50:
            categories.append('Fail')
        elif 50 <= score < 60:
            categories.append('Second Class')
        elif 60 <= score < 75:
            categories.append('First Class')
        else:
            categories.append('First Class with Distinction')
    return categories

def main():
    personal_factors_path = 'data/persoanl factor.csv'
    home_factors_path = 'data/home.csv'
    school_factors_path = 'data/school.csv'

    X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf = load_and_preprocess_data(personal_factors_path, 'Score')
    X_train_hf, X_test_hf, y_train_hf, y_test_hf, preprocessor_hf = load_and_preprocess_data(home_factors_path, 'Score')
    X_train_sf, X_test_sf, y_train_sf, y_test_sf, preprocessor_sf = load_and_preprocess_data(school_factors_path, 'Score')

    predictions_pf, y_test_pf = train_and_evaluate(X_train_pf, X_test_pf, y_train_pf, y_test_pf, preprocessor_pf)
    predictions_hf, y_test_hf = train_and_evaluate(X_train_hf, X_test_hf, y_train_hf, y_test_hf, preprocessor_hf)
    predictions_sf, y_test_sf = train_and_evaluate(X_train_sf, X_test_sf, y_train_sf, y_test_sf, preprocessor_sf)

    final_predictions = np.maximum.reduce([predictions_pf, predictions_hf, predictions_sf])
    #final_predictions = np.maximum.reduce([predictions_pf])

    categories = categorize_scores(final_predictions)

    for actual, predicted, category in zip(y_test_pf, final_predictions, categories):
        print(f"Actual: {actual}, Predicted: {predicted}, Category: {category}")

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Actual Score': list(y_test_pf),
        'Predicted Score': final_predictions,
        'Category': categories
    })

    # Export the DataFrame to an Excel file
    results_df.to_excel('model_predictions.xlsx', index=False)

    print("Results have been exported to 'model_predictions.xlsx'.")

if __name__ == "__main__":
    main()
