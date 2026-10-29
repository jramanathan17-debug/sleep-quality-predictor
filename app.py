
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# 1. Load trained ML model
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/sleep_quality_model.pkl")


model = load_model()

# ---------------------------------
# 2. Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Sleep Quality Predictor",
    page_icon="🌙",
    layout="wide"
)

# ---------------------------------
# 3. Styling
# ---------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #edf2ff, #f5edff);
}
h1 {
    color: #353477;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #555577;
    font-size: 18px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #5146a5;
    color: white;
    border-radius: 10px;
    width: 100%;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# 4. Header
# ---------------------------------
st.title("🌙 Sleep Quality Predictor")

st.markdown(
    '<p class="subtitle">Your personal sleep wellness assistant</p>',
    unsafe_allow_html=True
)

st.write(
    "Predict your sleep quality and explore food, yoga, "
    "reading and lifestyle suggestions."
)

st.divider()

# ---------------------------------
# 5. User input form
# ---------------------------------
st.subheader("📝 Enter Your Sleep Details")

with st.form("sleep_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        age = st.number_input(
            "Age", min_value=18, max_value=100, value=22
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Software Engineer", "Doctor", "Teacher",
                "Nurse", "Engineer", "Accountant",
                "Sales Representative", "Scientist",
                "Lawyer", "Manager", "Other"
            ]
        )

        sleep_duration = st.slider(
            "Sleep Duration (hours)",
            3.0, 12.0, 7.0, 0.1
        )

        physical_activity = st.slider(
            "Physical Activity (minutes/day)",
            0, 120, 30
        )

        stress_level = st.slider(
            "Stress Level (1–10)", 1, 10, 5
        )

    with col2:
        bmi_category = st.selectbox(
            "BMI Category",
            ["Normal", "Normal Weight", "Overweight", "Obese"]
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure", 80, 220, 120
        )

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure", 40, 140, 80
        )

        heart_rate = st.number_input(
            "Heart Rate (bpm)", 40, 180, 72
        )

        daily_steps = st.number_input(
            "Daily Steps", 0, 50000, 6000, step=500
        )

        sleep_disorder = st.selectbox(
            "Known Sleep Disorder",
            ["None", "Sleep Apnea", "Insomnia"]
        )

    submitted = st.form_submit_button(
        "🔮 Predict Sleep Quality"
    )

# ---------------------------------
# 6. Prediction
# ---------------------------------
if submitted:

    blood_pressure = f"{systolic_bp}/{diastolic_bp}"

    disorder_value = (
        None if sleep_disorder == "None"
        else sleep_disorder
    )

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Physical Activity Level": physical_activity,
        "Stress Level": stress_level,
        "BMI Category": bmi_category,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Sleep Disorder": disorder_value
    }])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.header("📊 Your Results")

    if prediction == "Good":
        st.success("🌟 Predicted Sleep Quality: GOOD")
        st.write(
            "Your details match patterns associated with good "
            "sleep quality in the training dataset."
        )

    elif prediction == "Average":
        st.warning("😴 Predicted Sleep Quality: AVERAGE")
        st.write(
            "Some of your details match patterns associated "
            "with average sleep quality in the training dataset."
        )

    else:
        st.error("💤 Predicted Sleep Quality: POOR")
        st.write(
            "Your details match patterns associated with poor "
            "sleep quality in the training dataset."
        )

    # ---------------------------------
    # 7. Possible concerns and solutions
    # ---------------------------------
    st.header("💡 Personalized Sleep Suggestions")

    if sleep_duration < 7:
        st.markdown("""
        **😴 Sleep duration**

        Regularly getting too little sleep may contribute to
        tiredness, difficulty concentrating and mood changes.

        **What you can try:** Keep a regular bedtime and wake-up
        time. Most adults generally need around 7–9 hours of sleep.
        """)
    elif sleep_duration > 9:
        st.markdown("""
        **🛌 Longer sleep duration**

        Sleep needs vary from person to person.

        **What you can try:** If you regularly sleep for long
        periods and still feel tired, consider seeking medical advice.
        """)
    else:
        st.markdown("""
        **✅ Sleep routine**

        Your reported duration is within the usual 7–9 hour range
        for most adults. Try to maintain a consistent sleep schedule.
        """)

    if stress_level >= 7:
        st.markdown("""
        **🧠 Stress management**

        **What you can try:** Take short breaks, practise gentle
        breathing, and create a relaxing routine before bedtime.
        """)
    elif stress_level <= 3:
        st.markdown("""
        **🌿 Relaxation**

        Continue healthy ways of managing stress and make time
        for activities that help you unwind.
        """)

    if physical_activity < 30:
        st.markdown("""
        **🚶 Daily activity**

        If appropriate for you, gradually add comfortable activities
        such as walking or gentle stretching during the day.
        """)

    # ---------------------------------
    # 8. Food recommendations
    # ---------------------------------
    st.divider()
    st.header("🥗 Sleep-Friendly Food Ideas")

    food_col1, food_col2 = st.columns(2)

    with food_col1:
        st.subheader("🌙 Evening options")
        st.markdown("""
        - A small bowl of oatmeal or porridge
        - A banana or a small portion of fruit
        - A small handful of nuts, if suitable for you
        - Warm milk, if you tolerate dairy
        - A light, balanced dinner that suits your needs
        """)

    with food_col2:
        st.subheader("⚠️ Habits to consider")
        st.markdown("""
        - Avoid heavy meals close to bedtime if they disturb sleep.
        - Reduce coffee, strong tea and other caffeine later in the day.
        - Avoid using alcohol as a way to fall asleep.
        - Drink enough water during the day; reduce late fluids
          if nighttime bathroom trips interrupt your sleep.
        """)

    st.caption(
        "These are general food ideas, not a treatment for sleep "
        "problems. Consider allergies, dietary needs and medical advice."
    )

    # ---------------------------------
    # 9. Gentle yoga and breathing
    # ---------------------------------
    st.divider()
    st.header("🧘 Gentle Yoga & Relaxation")

    st.write(
        "Choose comfortable, gentle movements. Stop if you feel "
        "pain, dizziness or discomfort."
    )

    with st.expander("1. Slow breathing — 3 to 5 minutes"):
        st.write("""
        Sit or lie comfortably. Breathe in gently and breathe out
        slowly without forcing your breath. Repeat at a comfortable pace.
        """)

    with st.expander("2. Gentle seated stretch — 1 to 2 minutes"):
        st.write("""
        Sit comfortably and gently stretch your shoulders and neck
        within a pain-free range. Avoid sudden movements.
        """)

    with st.expander("3. Legs-up-the-wall pose — optional"):
        st.write("""
        If comfortable and suitable for you, lie near a wall and rest
        your legs supported by it briefly. Skip this pose if it causes
        discomfort or is unsuitable for your health.
        """)

    with st.expander("4. Gentle relaxation pose — 3 minutes"):
        st.write("""
        Lie down or sit with support. Relax your shoulders and jaw,
        and let your breathing return to its natural rhythm.
        """)

    st.caption(
        "Yoga and breathing may help some people relax, but they do "
        "not replace medical treatment for sleep disorders."
    )

    # ---------------------------------
    # 10. Reading suggestions
    # ---------------------------------
    st.divider()
    st.header("📚 Bedtime Reading Suggestions")

    book_col1, book_col2 = st.columns(2)

    with book_col1:
        st.subheader("🌼 For a calm mind")
        st.markdown("""
        **The Comfort Book — Matt Haig**

        Short reflections that some readers may find comforting.

        **The Little Prince — Antoine de Saint-Exupéry**

        A short, thoughtful story that can be a gentle reading choice.
        """)

    with book_col2:
        st.subheader("🌱 For better habits")
        st.markdown("""
        **Atomic Habits — James Clear**

        Ideas for building small, consistent daily habits.

        **Ikigai — Héctor García and Francesc Miralles**

        Reflections on purpose and everyday life.
        """)

    st.info(
        "Choose a book you personally find relaxing. Try reading "
        "a paper book for 10–15 minutes before bed, with comfortable "
        "lighting. Stop reading if it makes you more alert."
    )

    # ---------------------------------
    # 11. Personalized lifestyle plan
    # ---------------------------------
    st.divider()
    st.header("🌙 Your Simple Night Routine")

    st.markdown("""
    1. **Before bedtime:** Dim bright lights and reduce screen use.
    2. **Prepare:** Finish your usual evening tasks and make your
       bedroom comfortable.
    3. **Relax:** Try gentle breathing or a comfortable stretch.
    4. **Read:** Spend 10–15 minutes with a calming book.
    5. **Sleep:** Aim for a consistent bedtime and wake-up time.
    """)

    if stress_level >= 7 or sleep_duration < 7:
        st.warning(
            "Start with one or two manageable changes instead of "
            "trying to change every habit at once."
        )
    else:
        st.success(
            "Keep following a consistent routine and review how "
            "rested you feel during the day."
        )

    # ---------------------------------
    # 12. Medical disclaimer
    # ---------------------------------
    st.divider()

    st.caption("""
    Important: Sleep quality predictions are based on patterns
    learned from a dataset. Food, yoga, book and lifestyle ideas are
    general wellness suggestions, not medical treatment. The app
    cannot determine whether a disease will occur. Seek advice from
    a qualified healthcare professional for persistent sleep problems,
    severe daytime sleepiness, loud snoring or breathing interruptions.
    """)

# ---------------------------------
# 13. Reset / refresh
# ---------------------------------
if st.button("🔄 Reset / Refresh"):
    st.rerun()