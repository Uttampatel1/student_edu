from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load pre-trained models
model_paths = {
    'gradient_boosting': 'models/best_model_gradient_boosting.joblib',
    'random_forest': 'models/best_model_random_forest.joblib',
    'svr': 'models/best_model_svr.joblib',
    'decision_tree': 'models/best_model_decision_tree.joblib',
    'elastic_net': 'models/best_model_elastic_net.joblib',
    'ridge': 'models/best_model_ridge.joblib'
}

models = {name: joblib.load(path) for name, path in model_paths.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Extract values from the form data and convert to a DataFrame
        features = [
            data.get('age', np.nan),
            data.get('gender', np.nan),
            data.get('family_type', np.nan),
            data.get('literacy', np.nan),
            data.get('income', np.nan),
            data.get('fathers_occupation', np.nan),
            data.get('residential_area', np.nan),
            data.get('school_type', np.nan),
            data.get('transportation', np.nan),
            data.get('Are you attending school regularly?', np.nan),
            data.get('Do you bunk class regularly?', np.nan),
            data.get('Are you feel fear to go to school?', np.nan),
            data.get('Do you have good teacher-students relationship??', np.nan),
            data.get('Do you suffer from unhealthy lifestyle?', np.nan),
            data.get('Do you interested to do your educational work at home?', np.nan),
            data.get('Do you go to tuition class or personal classes?', np.nan),
            data.get('Do you worried about financial condition of your family?', np.nan),
            data.get('Do your parents get personal involvement in your development?', np.nan),
            data.get('residency_satisfaction', np.nan),
            data.get('parent_contribution_satisfaction', np.nan),
            data.get('transportation_satisfaction', np.nan),
            data.get('what do you feel by society pressure in educational performance?', np.nan),
            data.get('What do you feel by family pressure in educational performance?', np.nan),
            data.get('urban_residency_satisfaction', np.nan),
            data.get('rural_residency_satisfaction', np.nan),
            data.get('home_atmosphere', np.nan),
            data.get('favorite_school_activity', np.nan),
            data.get('teacher_student_relationship', np.nan),
            data.get('Do you feel performance pressure in examination?', np.nan),
            data.get(' Do you feel fear to examination?', np.nan),
            data.get('Do your teachers get personal involvement in your development?', np.nan),
            data.get('school_infrastructure_satisfaction', np.nan),
            data.get('teacher_contribution_satisfaction', np.nan),
            data.get('school_transportation_satisfaction', np.nan),
            data.get('teacher_student_relationship_satisfaction', np.nan),
            data.get('rural_school_satisfaction', np.nan),
            data.get('urban_school_satisfaction', np.nan),
            data.get('school_environment_satisfaction', np.nan)
        ]

        df = pd.DataFrame([features], columns=[
            'age', 'gender', 'family_type', 'literacy', 'income', 'fathers_occupation', 'residential_area',
            'school_type', 'transportation', 'Are you attending school regularly?', 'Do you bunk class regularly?',
            'Are you feel fear to go to school?', 'Do you have good teacher-students relationship??',
            'Do you suffer from unhealthy lifestyle?', 'Do you interested to do your educational work at home?',
            'Do you go to tuition class or personal classes?', 'Do you worried about financial condition of your family?',
            'Do your parents get personal involvement in your development?', 'residency_satisfaction',
            'parent_contribution_satisfaction', 'transportation_satisfaction',
            'what do you feel by society pressure in educational performance?',
            'What do you feel by family pressure in educational performance?', 'urban_residency_satisfaction',
            'rural_residency_satisfaction', 'home_atmosphere', 'favorite_school_activity',
            'teacher_student_relationship', 'Do you feel performance pressure in examination?',
            ' Do you feel fear to examination?', 'Do your teachers get personal involvement in your development?',
            'school_infrastructure_satisfaction', 'teacher_contribution_satisfaction',
            'school_transportation_satisfaction', 'teacher_student_relationship_satisfaction',
            'rural_school_satisfaction', 'urban_school_satisfaction', 'school_environment_satisfaction'
        ])

        # Normalize the numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        min_vals = df[numeric_cols].min()
        max_vals = df[numeric_cols].max()
        df[numeric_cols] = (df[numeric_cols] - min_vals) / (max_vals - min_vals)

        # Convert boolean columns to integers (0 and 1)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

        # Apply label encoding for object columns
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = label_encoder.fit_transform(df[col])
        df['Normalized_Score'] = 0.5

        # Select model for prediction (you can add logic to choose different models)
        model = models['gradient_boosting']

        # Make prediction
        score = model.predict(df)[0]

        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
