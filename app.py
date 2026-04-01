from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import datetime
import os

# Initialize Flask app
app = Flask(__name__)

# Load and prepare dataset safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    df = pd.read_csv(os.path.join(BASE_DIR, 'final.csv'))
    print("Dataset loaded successfully")
except Exception as e:
    print("ERROR loading dataset:", e)
    # fallback dataset (prevents crash)
    df = pd.DataFrame({
        "Cleaned_text": ["test symptom"],
        "Category": ["medical"],
        "Health": ["General"]
    })

# Clean missing values
df['Cleaned_text'] = df['Cleaned_text'].fillna('')
df['Health'] = df['Health'].fillna('')

# -------- MODEL TRAINING --------

# Reduce dataset size (IMPORTANT for Render free tier)
df_subset = df.head(2000)

# Category model
independent_category = df_subset.Cleaned_text
dependent_category = df_subset.Category
in_train_cat, _, de_train_cat, _ = train_test_split(
    independent_category, dependent_category, test_size=0.3
)

# Health model
independent_health = df_subset.Cleaned_text
dependent_health = df_subset.Health
in_train_health, _, de_train_health, _ = train_test_split(
    independent_health, dependent_health, test_size=0.3
)

# Create models
tf_category = TfidfVectorizer()
lr_category = LogisticRegression(solver="sag", max_iter=200)
model_category = Pipeline([
    ("vectorizer", tf_category),
    ("classifier", lr_category)
])

tf_health = TfidfVectorizer()
lr_health = LogisticRegression(solver="sag", max_iter=200)
model_health = Pipeline([
    ("vectorizer", tf_health),
    ("classifier", lr_health)
])

# Train models
print("Training models...")
model_category.fit(in_train_cat, de_train_cat)
model_health.fit(in_train_health, de_train_health)
print("Models trained successfully")

# -------- DOCTOR DATA --------

doctors = {
    "General": "Dr. Smith",
    "Cardiology": "Dr. Emily",
    "Orthopedics": "Dr. Rahul",
    "Neurology": "Dr. Priya",
    "Dermatology": "Dr. Alice"
}

# In-memory appointment storage
appointments = []

# -------- ROUTES --------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('user_input', '')

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    try:
        result_category = model_category.predict([user_input])[0]

        if result_category.lower() == 'medical':
            result_health = model_health.predict([user_input])[0]

            doctor_info = doctors.get(
                result_health, "No doctor available"
            )

            response = (
                f"Classification Result: {result_category}\n"
                f"Health Specialty Prediction: {result_health}\n"
                f"Contact Doctor: {doctor_info}"
            )
        else:
            response = f"Classification Result: {result_category}"

    except Exception as e:
        response = f"Error processing request: {str(e)}"

    return jsonify({"response": response})


@app.route('/contact_doctor', methods=['POST'])
def contact_doctor():
    specialty = request.form.get('specialty', '')

    response = (
        f"You've contacted the doctor for {specialty}. "
        f"Doctor says: 'Please provide detailed symptoms.'"
    )

    return jsonify({"response": response})


@app.route('/schedule_appointment', methods=['POST'])
def schedule_appointment():
    data = request.get_json()

    specialty = data.get('specialty', '')
    user_name = data.get('user_name', '')
    preferred_date = data.get('preferred_date', '')

    if not specialty or not user_name or not preferred_date:
        return jsonify({
            "response": "Missing required information."
        }), 400

    try:
        appointment_date = datetime.datetime.strptime(
            preferred_date, '%Y-%m-%d'
        ).date()

        if appointment_date < datetime.date.today():
            return jsonify({
                "response": "Cannot schedule in the past."
            }), 400

        appointments.append({
            "user_name": user_name,
            "specialty": specialty,
            "appointment_date": appointment_date.strftime('%Y-%m-%d')
        })

        return jsonify({
            "response": f"Appointment scheduled for {user_name} on {appointment_date}"
        })

    except ValueError:
        return jsonify({
            "response": "Invalid date format (YYYY-MM-DD)."
        }), 400


@app.route('/schedule', methods=['GET', 'POST'])
def schedule():
    global appointments

    if request.method == 'POST':
        name = request.form.get('name')
        date = request.form.get('date')
        speciality = request.form.get('speciality')
        reason = request.form.get('reason')

        appointments.append({
            'name': name,
            'date': date,
            'speciality': speciality,
            'reason': reason
        })

    return render_template('schedule.html', appointments=appointments)


# -------- MAIN --------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
#cd pred
#pip install scikit-learn
#python app.py
