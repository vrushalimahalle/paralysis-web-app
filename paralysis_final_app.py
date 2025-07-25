import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
from fpdf import FPDF
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ------------------ Config ------------------
st.set_page_config(page_title="Paralysis Predictor", layout="wide")
os.makedirs("saved_reports", exist_ok=True)

# ------------------ Login ------------------
users = {"admin": "admin123", "doctor": "doc123"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.rerun()  # ✅ updated line
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------ Dataset & Model ------------------
def generate_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(20, 80, 500),
        'Numbness': np.random.randint(0, 2, 500),
        'Weakness': np.random.randint(0, 2, 500),
        'Confusion': np.random.randint(0, 2, 500),
        'Stiffness': np.random.randint(0, 2, 500),
        'Spasms': np.random.randint(0, 2, 500),
        'Cramps': np.random.randint(0, 2, 500),
        'Atrophy': np.random.randint(0, 2, 500),
        'Speech': np.random.randint(0, 2, 500),
        'Walking': np.random.randint(0, 2, 500),
        'Dizzy': np.random.randint(0, 2, 500),
        'Balance': np.random.randint(0, 2, 500),
        'Headache': np.random.randint(0, 2, 500),
    })
    df['Paralysis'] = np.random.randint(0, 2, 500)
    types = ['Monoplegia', 'Diplegia', 'Hemiplegia', 'Paraplegia', 'Quadriplegia']
    df['Paralysis_Type'] = np.where(df['Paralysis'] == 1, np.random.choice(types, 500), 'None')
    return df

data = generate_data()
X = data.iloc[:, :-2]
y = data['Paralysis']
type_y = data['Paralysis_Type']

risk_model = RandomForestClassifier(n_estimators=100)
risk_model.fit(X, y)
type_model = RandomForestClassifier(n_estimators=100)
type_model.fit(X[y == 1], type_y[y == 1])

# ------------------ Email Function ------------------
def send_email(to_email, subject, body, attachment_path=None):
    from_email = st.secrets["EMAIL"]
    password = st.secrets["PASSWORD"]

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachment_path:
        with open(attachment_path, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachment_path))
            msg.attach(attach)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)

# ------------------ Prediction Form ------------------
st.title("🧠 Paralysis Disease Prediction")

with st.form("form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100)
        email = st.text_input("Email")
    with col2:
        phone = st.text_input("Phone")
        address = st.text_area("Address")
        remarks = st.text_area("Doctor's Remarks")

    st.markdown("### Symptoms")
    cols = st.columns(3)
    symptoms = [
        "Numbness", "Weakness", "Confusion", "Stiffness",
        "Spasms", "Cramps", "Atrophy", "Speech",
        "Walking", "Dizzy", "Balance", "Headache"
    ]
    input_values = []
    for i, sym in enumerate(symptoms):
        input_values.append(cols[i % 3].checkbox(sym))

    submitted = st.form_submit_button("🔍 Predict")

# ------------------ Prediction ------------------
if submitted:
    input_df = pd.DataFrame([{
        "Age": age, **{symptoms[i]: int(val) for i, val in enumerate(input_values)}
    }])

    risk_pred = risk_model.predict(input_df)[0]
    risk_proba = risk_model.predict_proba(input_df)[0][1]
    paralysis_type = type_model.predict(input_df)[0] if risk_pred == 1 else "None"

    st.subheader("🧾 Result")
    if risk_pred == 1:
        st.error(f"⚠ High Risk of Paralysis Detected")
        st.warning(f"Predicted Type: {paralysis_type}")
        st.info(f"Confidence: {risk_proba*100:.2f}%")
    else:
        st.success("✅ Low Risk of Paralysis")
        st.info(f"Confidence: {(1-risk_proba)*100:.2f}%")

    # ------------------ PDF ------------------
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{name.replace(' ', '_')}_{now}.pdf"
    filepath = os.path.join("saved_reports", filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Paralysis Disease Prediction Report", ln=1, align="C")
    pdf.cell(200, 10, f"Name: {name}", ln=1)
    pdf.cell(200, 10, f"Gender: {gender}", ln=1)
    pdf.cell(200, 10, f"Age: {age}", ln=1)
    pdf.cell(200, 10, f"Email: {email}", ln=1)
    pdf.cell(200, 10, f"Phone: {phone}", ln=1)
    pdf.cell(200, 10, f"Address: {address}", ln=1)
    pdf.cell(200, 10, f"Risk: {'High' if risk_pred else 'Low'}", ln=1)
    pdf.cell(200, 10, f"Confidence: {risk_proba*100:.2f}%", ln=1)
    pdf.cell(200, 10, f"Type: {paralysis_type}", ln=1)
    pdf.multi_cell(0, 10, f"Doctor's Remarks: {remarks}")
    pdf.output(filepath)

    with open(filepath, "rb") as f:
        st.download_button("📄 Download Report PDF", f, file_name=filename)

    # ------------------ Save & Email ------------------
    row = input_df.copy()
    row["Name"] = name
    row["Gender"] = gender
    row["Email"] = email
    row["Phone"] = phone
    row["Risk"] = 'High' if risk_pred else 'Low'
    row["Confidence"] = f"{risk_proba*100:.2f}%"
    row["Type"] = paralysis_type
    row["Timestamp"] = now

    if os.path.exists("patients.csv"):
        df = pd.read_csv("patients.csv")
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv("patients.csv", index=False)

    # Send email
    try:
        send_email(email, "Paralysis Prediction Report", "Please find attached your report.", filepath)
        send_email("admin@example.com", f"Patient: {name}", "New report received", filepath)
        st.success("📧 Report emailed successfully")
    except:
        st.warning("⚠ Could not send email. Check credentials in secrets")

# ------------------ Admin Panel ------------------
st.markdown("---")
st.header("📊 Admin Panel")
tabs = st.tabs(["📂 Patients", "📈 Graphs", "📦 ZIP Download"])

with tabs[0]:
    if os.path.exists("patients.csv"):
        df = pd.read_csv("patients.csv")
        search = st.text_input("Search by name")
        if search:
            df = df[df["Name"].str.contains(search, case=False)]
        st.dataframe(df)

with tabs[1]:
    if os.path.exists("patients.csv"):
        df = pd.read_csv("patients.csv")
        if "Risk" in df:
            st.plotly_chart(px.pie(df, names="Risk", title="Risk Distribution"))
        if "Type" in df:
            st.bar_chart(df["Type"].value_counts())

with tabs[2]:
    if os.path.exists("saved_reports"):
        zip_name = "All_Reports.zip"
        with zipfile.ZipFile(zip_name, "w") as z:
            for f in os.listdir("saved_reports"):
                z.write(os.path.join("saved_reports", f), f)
        with open(zip_name, "rb") as f:
            st.download_button("⬇ Download All Reports", f, file_name=zip_name)
