from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import pickle
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "ml_project_secret"

# -------------------------------
# Load trained ML model
# -------------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model.pkl. Risk calculation will use fallback. Error: {e}")
    model = None

# -------------------------------
# Database helper
# -------------------------------
def get_db():
    conn = sqlite3.connect("database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, first_name TEXT, last_name TEXT)''')
    db.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (username TEXT, symptoms TEXT, age INTEGER, sex TEXT, 
                  disease TEXT, risk REAL, date TIMESTAMP)''')
    db.commit()

init_db()

def login_required():
    return 'user' in session

def recommendation(risk):
    if risk < 30:
        return "Low risk. Maintain a healthy lifestyle and rest."
    elif risk < 60:
        return "Moderate risk. Please monitor symptoms and consult a doctor if they persist."
    else:
        return "High risk. Immediate medical attention is highly recommended."

# -------------------------------
# Disease inference logic
# -------------------------------
def infer_disease(symptoms):
    s = symptoms.lower()
    if any(x in s for x in ["blood", "tb", "tuberculosis", "night sweats"]):
        return "tuberculosis"
    if any(x in s for x in ["cold", "sneezing", "runny nose", "sore throat"]):
        if not any(x in s for x in ["wheezing", "chest pain", "tightness"]):
            return "common cold"
    if any(x in s for x in ["wheezing", "asthma", "tightness"]):
        return "asthma"
    if "fever" in s and ("chest" in s or "breath" in s):
        return "pneumonia"
    if any(x in s for x in ["phlegm", "mucus", "bronchitis"]):
        return "bronchitis"
    if any(x in s for x in ["smoking", "copd", "chronic"]):
        return "copd"
    if any(x in s for x in ["fibrosis", "scarring"]):
        return "pulmonary fibrosis"
    return "common cold"

# -------------------------------
# DYNAMIC CONTENT DICTIONARIES
# -------------------------------
disease_info = {
    "asthma": {"description": "Asthma is a chronic condition causing airway inflammation.", "image": "asthma.png"},
    "bronchitis": {"description": "Bronchitis is inflammation of the bronchial tubes, often with mucus.", "image": "bronchitis.png"},
    "pneumonia": {"description": "Pneumonia is an infection that inflames the air sacs in the lungs.", "image": "pneumonia.png"},
    "copd": {"description": "COPD causes airflow blockage and breathing problems.", "image": "copd.png"},
    "tuberculosis": {"description": "TB is a serious bacterial disease affecting the lungs.", "image": "tuberculosis.png"},
    "pulmonary fibrosis": {"description": "Occurs when lung tissue becomes damaged and scarred.", "image": "fibrosis.png"},
    "common cold": {"description": "A mild viral infection of the nose and throat.", "image": "cold.png"}
}

# Unique recommendations for the Dashboard based on latest disease
disease_recommendations = {
    "asthma": ["Carry your inhaler at all times.", "Avoid cold air and dust triggers.", "Monitor peak flow daily."],
    "pneumonia": ["Complete full antibiotic course.", "Practice deep breathing exercises.", "Keep yourself warm and rested."],
    "tuberculosis": ["Strict medication adherence is vital.", "Wear a mask in crowded spaces.", "Maintain a high-protein diet."],
    "common cold": ["Stay hydrated with warm fluids.", "Get 8+ hours of sleep.", "Use saline nasal drops."],
    "bronchitis": ["Avoid smoking/secondhand smoke.", "Use a humidifier at night.", "Gargle with warm salt water."],
    "copd": ["Use oxygen therapy as prescribed.", "Avoid air pollutants/heavy dust.", "Practice pursed-lip breathing."],
    "pulmonary fibrosis": ["Avoid lung irritants.", "Stay updated on flu/pneumonia vaccines.", "Maintain mild physical activity."]
}

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html", page_class="home-page")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first, last = request.form['first_name'], request.form['last_name']
        user, pwd = request.form['username'], request.form['password']
        db = get_db()
        try:
            db.execute("INSERT INTO users VALUES (?,?,?,?)", (user, pwd, first, last))
            db.commit()
            return redirect('/login')
        except sqlite3.IntegrityError:
            return "Username already exists!"
    return render_template("signup.html", page_class="auth-page")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user, pwd = request.form['username'], request.form['password']
        db = get_db()
        res = db.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd)).fetchone()
        if res:
            session.update({'user': user, 'first_name': res['first_name'], 'last_name': res['last_name']})
            return redirect('/predict')
    return render_template("login.html", page_class="auth-page")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not login_required(): return redirect('/login')
    if request.method == 'POST':
        symptoms, age, sex = request.form['symptoms'], int(request.form['age']), request.form['sex']
        disease = infer_disease(symptoms)
        
        # Risk Logic
        input_df = pd.DataFrame([{"Symptoms": symptoms, "Age": age, "Sex": sex, "Disease": disease}])
        risk = 45.0
        if model:
            try: risk = model.predict_proba(input_df)[0][1] * 100
            except: pass
        if disease == "common cold": risk = min(risk, 22.0)

        db = get_db()
        db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)", 
                   (session['user'], symptoms, age, sex, disease, round(risk, 2), datetime.now().strftime("%Y-%m-%d %H:%M")))
        db.commit()
        
        info = disease_info.get(disease, disease_info["common cold"])
        return render_template("result.html", username=f"{session['first_name']} {session['last_name']}",
                               disease=disease.title(), description=info["description"], image=info["image"],
                               risk=round(risk, 2), advice=recommendation(risk), page_class="result-page")
    return render_template("predict.html", page_class="predict-page")

@app.route('/dashboard')
def dashboard():
    if not login_required(): return redirect('/login')
    db = get_db()
    
    # 1. FETCH LATEST DATA FOR REAL-TIME FEEL
    latest = db.execute("SELECT disease, risk FROM predictions WHERE username=? ORDER BY date DESC LIMIT 1", 
                        (session['user'],)).fetchone()
    
    current_risk = latest['risk'] if latest else 0
    current_disease = latest['disease'] if latest else "None"
    
    # 2. DYNAMIC ALERT LOGIC
    alert_status = "STABLE"
    if current_risk > 70: alert_status = "CRITICAL"
    elif current_risk > 35: alert_status = "MODERATE"

    # 3. DYNAMIC RECOMMENDATIONS BASED ON DISEASE
    tips = disease_recommendations.get(current_disease, ["Analyze symptoms to see personalized tips.", "Maintain healthy habits.", "Stay hydrated."])

    # 4. STATS
    total = db.execute("SELECT COUNT(*) FROM predictions WHERE username=?", (session['user'],)).fetchone()[0]
    avg = db.execute("SELECT AVG(risk) FROM predictions WHERE username=?", (session['user'],)).fetchone()[0] or 0
    history = db.execute("SELECT * FROM predictions WHERE username=? ORDER BY date DESC LIMIT 5", (session['user'],)).fetchall()

    return render_template("dashboard.html", total=total, avg=round(avg, 2), current_risk=current_risk,
                           current_disease=current_disease.title(), alert_status=alert_status, tips=tips,
                           first_name=session['first_name'], history=history, page_class="dashboard-page")

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)