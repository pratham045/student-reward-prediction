
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Reward Prediction System",
    page_icon="🎓",
    layout="wide"
)

model = joblib.load("student_reward_model.pkl")
df = pd.read_csv("Students Performance .csv")

st.title("🎓 Student Reward Prediction System")
st.write("Predict student grade/reward outcome based on student activities.")

st.divider()

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Students", len(df))
c2.metric("Features Used", 13)
c3.metric("Grade Categories", df["Grade"].nunique())

st.divider()

st.subheader("📈 Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Grade", ax=ax)
    ax.set_title("Grade Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Attendance", hue="Grade", ax=ax)
    ax.set_title("Attendance vs Grade")
    st.pyplot(fig)

st.divider()

st.subheader("🔮 Predict Student Outcome")

col1, col2, col3 = st.columns(3)

with col1:
    student_age = st.selectbox(
        "Student Age",
        sorted(df["Student_Age"].dropna().unique())
    )

    sex = st.selectbox(
        "Sex",
        sorted(df["Sex"].dropna().unique())
    )

    high_school = st.selectbox(
        "High School Type",
        sorted(df["High_School_Type"].dropna().unique())
    )

    scholarship = st.selectbox(
        "Scholarship",
        sorted(df["Scholarship"].dropna().unique())
    )

    additional_work = st.selectbox(
        "Additional Work",
        sorted(df["Additional_Work"].dropna().unique())
    )

with col2:
    sports_activity = st.selectbox(
        "Sports Activity",
        sorted(df["Sports_activity"].dropna().unique())
    )

    transportation = st.selectbox(
        "Transportation",
        sorted(df["Transportation"].dropna().unique())
    )

    weekly_study_hours = st.selectbox(
        "Weekly Study Hours",
        sorted(df["Weekly_Study_Hours"].dropna().unique())
    )

    attendance = st.selectbox(
        "Attendance",
        sorted(df["Attendance"].dropna().unique())
    )

    reading = st.selectbox(
        "Reading",
        sorted(df["Reading"].dropna().unique())
    )

with col3:
    notes = st.selectbox(
        "Notes",
        sorted(df["Notes"].dropna().unique())
    )

    listening = st.selectbox(
        "Listening in Class",
        sorted(df["Listening_in_Class"].dropna().unique())
    )

    project_work = st.selectbox(
        "Project Work",
        sorted(df["Project_work"].dropna().unique())
    )

if st.button("🎯 Predict Student Outcome", use_container_width=True):

    input_data = pd.DataFrame({
        "Student_Age": [student_age],
        "Sex": [sex],
        "High_School_Type": [high_school],
        "Scholarship": [scholarship],
        "Additional_Work": [additional_work],
        "Sports_activity": [sports_activity],
        "Transportation": [transportation],
        "Weekly_Study_Hours": [weekly_study_hours],
        "Attendance": [attendance],
        "Reading": [reading],
        "Notes": [notes],
        "Listening_in_Class": [listening],
        "Project_work": [project_work]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Student Grade / Outcome: {prediction}"
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        prob_df = pd.DataFrame({
            "Grade": model.classes_,
            "Probability": probabilities
        })

        prob_df = prob_df.sort_values(
            "Probability",
            ascending=False
        )

        st.subheader("Prediction Probability")
        st.dataframe(prob_df, use_container_width=True)

st.divider()

st.caption("Student Reward Prediction System")
