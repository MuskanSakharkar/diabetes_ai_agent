from flask import Flask, render_template, request
import pickle
import csv
from datetime import datetime

app = Flask(__name__)

# Load model and encoders safely
try:
    with open('model/diabetes_model.pkl', 'rb') as f:
        model, le_gender, le_smoke = pickle.load(f)
except Exception as e:
    print("Error loading model:", e)
    model, le_gender, le_smoke = None, None, None


@app.route('/')
def home():
    disclaimer = "⚠️ This system is for educational purposes only and is not a substitute for professional medical advice."
    return render_template('index.html', disclaimer=disclaimer)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template(
            'index.html',
            prediction="Error",
            risk_level="Unavailable",          # ✅ added
            confidence=0,                      # ✅ added
            recommendation="",                 # ✅ added
            explanation="Model not available.",
            disclaimer="⚠️ System unavailable",
            repeat_alert=None,
            agent_alert_title="",
            agent_reason_text="",
            immediate_advice=[],
            agent_explanation=""
        )

    try:
        # Encode categorical inputs
        gender = le_gender.transform([request.form['gender']])[0]
        smoking = le_smoke.transform([request.form['smoking']])[0]

        # Numerical inputs
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart = int(request.form['heart'])
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        glucose = float(request.form['glucose'])

        features = [[gender, age, hypertension, heart, smoking, bmi, hba1c, glucose]]

        # ML prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        confidence = round(probability * 100, 2) if prediction == 1 else round((1 - probability) * 100, 2)

        # ---------------- REASONING ----------------
        reasons = []
        if glucose > 140:
            reasons.append("high blood glucose level")
        if hba1c > 6.5:
            reasons.append("elevated HbA1c level")
        if bmi > 30:
            reasons.append("obesity (high BMI)")
        if hypertension == 1:
            reasons.append("presence of hypertension")
        if heart == 1:
            reasons.append("history of heart disease")

        # ---------------- RISK LEVEL ----------------
        if probability > 0.75:
            risk_level = "High"
        elif probability > 0.5:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # ---------------- RESULT ----------------
        if prediction == 1:
            result = "Diabetic ❌"
            explanation = "⚠️ Higher risk due to " + ", ".join(reasons) + "."
            recommendation = "Immediate medical consultation is recommended."
        else:
            result = "Not Diabetic ✅"
            explanation = "✅ Lower risk as key indicators are within safer ranges."
            recommendation = "Maintain a healthy lifestyle and regular checkups."

        # ---------------- AI AGENT ALERT ----------------
        agent_alert_title = ""
        agent_reason_text = ""
        immediate_advice = []

        if risk_level == "High":
            agent_alert_title = " High Alert  ⚠️"
            agent_reason_text = "Reason: " + " + ".join(reasons) if reasons else "Reason: Multiple risk factors detected"
            immediate_advice = [
                "Avoid sugary foods",
                "Drink water regularly",
                "Seek medical help within 24 hours"
            ]

        elif risk_level == "Moderate":
            agent_alert_title = "Moderate Alert ⚠️"
            agent_reason_text = "Reason: Some indicators are above normal range"
            immediate_advice = [
                "Reduce sugar intake",
                "Increase physical activity",
                "Monitor glucose levels weekly"
            ]

        else:
            agent_alert_title = "Low Alert"
            agent_reason_text = "Reason: Health indicators are within safer ranges"
            immediate_advice = [
                "Maintain a balanced diet",
                "Exercise regularly",
                "Continue routine checkups"
            ]

        # ---------------- AI AGENT EXPLANATION ----------------
        agent_explanation = (
            "🤖 AI Agent Analysis: The model evaluated glucose, HbA1c, BMI, age, "
            "smoking history, hypertension, and heart disease to assess diabetes risk."
        )

        # ---------------- TRACKING ----------------
        with open("health_logs.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), age, bmi, glucose, hba1c, risk_level])

        # ---------------- RENDER ----------------
        return render_template(
            'index.html',
            prediction=result,
            risk_level=risk_level,
            confidence=confidence,
            explanation=explanation,
            recommendation=recommendation,

            agent_alert_title=agent_alert_title,
            agent_reason_text=agent_reason_text,
            immediate_advice=immediate_advice,
            agent_explanation=agent_explanation,

            repeat_alert=None,
            disclaimer="⚠️ This system is for educational purposes only and not a medical diagnosis tool."
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction="Error",
            risk_level="Error",              # ✅ added
            confidence=0,                    # ✅ added
            recommendation="",               # ✅ added
            explanation=str(e),
            disclaimer="⚠️ Something went wrong",
            repeat_alert=None,
            agent_alert_title="",
            agent_reason_text="",
            immediate_advice=[],
            agent_explanation=""
        )


if __name__ == "__main__":
    app.run(debug=True)


