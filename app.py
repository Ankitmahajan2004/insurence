import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Safe Path Resolution for Vercel & Local Deployment ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_Regression_model.pkl")

model = None
model_error = None

# Safely load the regression model
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model_error = f"Model file not found at path: {MODEL_PATH}"
except Exception as e:
    model_error = f"Error loading model: {str(e)}"


# --- Responsive UI Layout & Inline Styling ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --input-bg: #1f2937;
            --border-color: #374151;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-glow: rgba(99, 102, 241, 0.25);
            --success-bg: rgba(16, 185, 129, 0.1);
            --success-border: #10b981;
            --radius: 16px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.12) 0px, transparent 50%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 16px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            max-width: 520px;
            width: 100%;
            padding: 40px 32px;
            border-radius: var(--radius);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .header h1 {
            color: var(--text-main);
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        .header p {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 8px;
        }

        .alert-error {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 13px;
            margin-bottom: 24px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
        }

        input, select {
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 14px;
            border-radius: 10px;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        input::placeholder {
            color: #6b7280;
        }

        button.submit-btn {
            grid-column: span 2;
            background: var(--accent-primary);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 12px;
            transition: all 0.2s ease;
        }

        button.submit-btn:hover {
            background: var(--accent-hover);
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 28px;
            background: var(--success-bg);
            border: 1px solid var(--success-border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .result-card span {
            display: block;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-card h2 {
            color: #10b981;
            font-size: 32px;
            font-weight: 800;
            margin-top: 4px;
        }

        @media (max-width: 480px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
            button.submit-btn {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>

<div class="card">
    <div class="header">
        <h1>Insurance Cost Predictor</h1>
        <p>Enter user demographics to calculate estimated medical insurance cost</p>
    </div>

    {% if model_error %}
        <div class="alert-error">
            <strong>Model Load Warning:</strong> {{ model_error }}
        </div>
    {% endif %}

    <form method="POST" action="/predict" class="form-grid">
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" placeholder="e.g. 28" required min="18" max="100" value="{{ form_data.age if form_data else '' }}">
        </div>

        <div class="form-group">
            <label for="sex">Gender</label>
            <select id="sex" name="sex" required>
                <option value="" disabled {% if not form_data %}selected{% endif %}>Select Gender</option>
                <option value="0" {% if form_data and form_data.sex == '0' %}selected{% endif %}>Female (0)</option>
                <option value="1" {% if form_data and form_data.sex == '1' %}selected{% endif %}>Male (1)</option>
            </select>
        </div>

        <div class="form-group">
            <label for="bmi">BMI Index</label>
            <input type="number" step="0.1" id="bmi" name="bmi" placeholder="e.g. 24.5" required value="{{ form_data.bmi if form_data else '' }}">
        </div>

        <div class="form-group">
            <label for="children">Children</label>
            <input type="number" id="children" name="children" placeholder="e.g. 0" required min="0" max="10" value="{{ form_data.children if form_data else '' }}">
        </div>

        <div class="form-group">
            <label for="smoker">Smoker</label>
            <select id="smoker" name="smoker" required>
                <option value="" disabled {% if not form_data %}selected{% endif %}>Select Option</option>
                <option value="0" {% if form_data and form_data.smoker == '0' %}selected{% endif %}>No (0)</option>
                <option value="1" {% if form_data and form_data.smoker == '1' %}selected{% endif %}>Yes (1)</option>
            </select>
        </div>

        <div class="form-group">
            <label for="region">Region Code</label>
            <select id="region" name="region" required>
                <option value="" disabled {% if not form_data %}selected{% endif %}>Select Region</option>
                <option value="0" {% if form_data and form_data.region == '0' %}selected{% endif %}>Southwest (0)</option>
                <option value="1" {% if form_data and form_data.region == '1' %}selected{% endif %}>Southeast (1)</option>
                <option value="2" {% if form_data and form_data.region == '2' %}selected{% endif %}>Northwest (2)</option>
                <option value="3" {% if form_data and form_data.region == '3' %}selected{% endif %}>Northeast (3)</option>
            </select>
        </div>

        <button type="submit" class="submit-btn" {% if model_error %}disabled style="opacity:0.5; cursor:not-allowed;"{% endif %}>
            Calculate Estimate
        </button>
    </form>

    {% if prediction is not none %}
        <div class="result-card">
            <span>Estimated Insurance Cost</span>
            <h2>${{ "%.2f"|format(prediction) }}</h2>
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, model_error=model_error, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            model_error=f"Model not loaded correctly. Check server logs.",
            prediction=None
        ), 500

    try:
        # Features in required order: age, sex, bmi, children, smoker, region
        age = float(request.form.get("age", 0))
        sex = float(request.form.get("sex", 0))
        bmi = float(request.form.get("bmi", 0))
        children = float(request.form.get("children", 0))
        smoker = float(request.form.get("smoker", 0))
        region = float(request.form.get("region", 0))

        # Shape array: (1, 6)
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Predict regression continuous float value
        prediction_val = float(model.predict(features)[0])

        return render_template_string(
            HTML_TEMPLATE,
            prediction=prediction_val,
            form_data=request.form,
            model_error=model_error
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            model_error=f"Prediction Failed: {str(e)}",
            form_data=request.form,
            prediction=None
        ), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
