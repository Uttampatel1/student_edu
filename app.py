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
            data.get('attend_school', np.nan),
            data.get('bunk_class', np.nan),
            data.get('fear_school', np.nan),
            data.get('relationship_with_parents', np.nan),
            data.get('unhealthy_lifestyle', np.nan),
            data.get('distracted_by_gadgets', np.nan),
            data.get('educational_work_at_home', np.nan),
            data.get('tuition_classes', np.nan),
            data.get('financial_worry', np.nan),
            data.get('parental_involvement', np.nan),
            data.get('residency_satisfaction', np.nan),
            data.get('parent_contribution_satisfaction', np.nan),
            data.get('transportation_satisfaction', np.nan),
            data.get('society_pressure', np.nan),
            data.get('family_pressure', np.nan),
            data.get('urban_residency_satisfaction', np.nan),
            data.get('rural_residency_satisfaction', np.nan),
            data.get('home_atmosphere', np.nan),
            data.get('favorite_school_activity', np.nan),
            data.get('teacher_student_relationship', np.nan),
            data.get('exam_performance_pressure', np.nan),
            data.get('exam_fear', np.nan),
            data.get('teacher_involvement', np.nan),
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
            'school_type', 'transportation', 'attend_school', 'bunk_class', 'fear_school', 'relationship_with_parents',
            'unhealthy_lifestyle', 'distracted_by_gadgets', 'educational_work_at_home', 'tuition_classes',
            'financial_worry', 'parental_involvement', 'residency_satisfaction', 'parent_contribution_satisfaction',
            'transportation_satisfaction', 'society_pressure', 'family_pressure', 'urban_residency_satisfaction',
            'rural_residency_satisfaction', 'home_atmosphere', 'favorite_school_activity', 'teacher_student_relationship',
            'exam_performance_pressure', 'exam_fear', 'teacher_involvement', 'school_infrastructure_satisfaction',
            'teacher_contribution_satisfaction', 'school_transportation_satisfaction',
            'teacher_student_relationship_satisfaction', 'rural_school_satisfaction', 'urban_school_satisfaction',
            'school_environment_satisfaction'
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

        # Select model for prediction (you can add logic to choose different models)
        model = models['gradient_boosting']

        # Make prediction
        score = model.predict(df)[0]

        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0')
