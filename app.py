import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(page_title="FitTrack Pro", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏋️ Personal Fitness Tracker Pro")
st.write("Transform your fitness journey with AI-powered insights!")

# How-to-use guide
with st.expander("ℹ️ How to Use FitTrack Pro"):
    st.write("""
    1. Adjust your profile settings in the sidebar.
    2. See your predicted calories burned and progress.
    3. Check your achievements and get personalized recommendations.
    4. Explore community comparisons and workout plans.
    """)

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Your Fitness Profile")
    age = st.slider("Age", 15, 80, 30)
    gender = st.radio("Gender", ("Male", "Female"))
    height = st.slider("Height (cm)", 140, 210, 170)
    weight = st.slider("Weight (kg)", 40, 150, 70)
    duration = st.slider("Workout Duration (min)", 10, 120, 30)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120)
    body_temp = st.slider("Body Temp (°C)", 36.0, 42.0, 37.5)
    target_calories = st.number_input("Daily Calorie Goal (kcal)", 100, 1000, 300)
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    st.metric("BMI", f"{bmi:.1f}")

# Data Processing
@st.cache_data
def load_data():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = exercise.merge(calories, on="User_ID")
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    return df

df = load_data()

# Model Training
X = df[['Gender', 'Age', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp']]
X.loc[:, 'Gender'] = X['Gender'].map({'male': 1, 'female': 0})
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)

# Prepare user input
user_data = pd.DataFrame([[1 if gender == 'Male' else 0, age, bmi, duration, heart_rate, body_temp]],
                         columns=['Gender', 'Age', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp'])
calories_burned = model.predict(user_data)[0]

# Initialize session state for workout history
if 'workouts' not in st.session_state:
    st.session_state.workouts = []

# Save workout button
if st.sidebar.button("💾 Save Workout"):
    workout = {
        'calories': calories_burned,
        'duration': duration,
        'heart_rate': heart_rate,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.workouts.append(workout)
    st.sidebar.success("Workout saved!")

# Main Dashboard
col1, col2 = st.columns(2)

with col1:
    # Calories Burned Prediction
    st.subheader("🔥 Calories Burned Prediction")
    st.markdown(f"<h1 style='text-align: center; color: #ff4b4b;'>{calories_burned:.0f} kcal</h1>", 
                unsafe_allow_html=True)
    
    # Progress bar for daily goal
    progress = min(calories_burned / target_calories, 1.0)
    st.progress(progress)
    st.write(f"{calories_burned:.0f} / {target_calories} kcal burned today")
    
    # Progress Visualization
    st.subheader("📊 Your Progress")
    fig = px.histogram(df, x='Calories', marginal="rug", title="Calories Burned Distribution")
    fig.add_vline(x=calories_burned, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Achievements
    st.subheader("🏆 Achievements")
    badges = []
    if calories_burned > 300:
        badges.append("🔥 Burn 300+ kcal")
    if duration > 45:
        badges.append("⏱️ Marathon Runner")
    if heart_rate > 160:
        badges.append("💓 High Intensity")
    
    if badges:
        for badge in badges:
            st.success(f"🌟 {badge}")
    else:
        st.info("🏁 Start training to unlock achievements!")

# Personalized Recommendations
st.subheader("💡 Personalized Recommendations")
rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    if bmi > 25:
        st.error("📉 Consider weight management exercises")
    elif bmi < 18.5:
        st.warning("📈 Focus on muscle-building workouts")
    else:
        st.success("✅ Your BMI is in healthy range!")

with rec_col2:
    if heart_rate > 160:
        st.error("⚠️ High heart rate! Consider rest")
    elif heart_rate < 100:
        st.warning("💪 Push harder next session!")
    else:
        st.success("🎯 Ideal workout intensity!")

with rec_col3:
    if duration < 30:
        st.warning("⏳ Try extending workout duration")
    else:
        st.success("🕒 Great workout duration!")

# Community Comparison
st.subheader("📈 Community Comparison")
tab1, tab2, tab3 = st.tabs(["BMI Analysis", "Age Trends", "Heart Rate Impact"])

with tab1:
    fig = px.scatter(df, x='BMI', y='Calories', color='Gender', title="BMI vs Calories Burned")
    fig.add_scatter(x=[bmi], y=[calories_burned], mode='markers', 
                    marker=dict(color='red', size=12, symbol='star'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(df.groupby('Age')['Calories'].mean().reset_index(), 
                  x='Age', y='Calories', title="Average Calories by Age")
    fig.add_vline(x=age, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.scatter(df, x='Heart_Rate', y='Calories', trendline="ols", title="Heart Rate vs Calories")
    st.plotly_chart(fig, use_container_width=True)

# Recommended Workouts
st.subheader("🚀 Recommended Workouts")
if calories_burned < 200:
    st.write("""
    **Light Exercise Plan:**
    - 🚶 30 min brisk walking
    - 🧘 15 min yoga session
    - 🤸 10 min stretching
    """)
elif 200 <= calories_burned < 400:
    st.write("""
    **Moderate Exercise Plan:**
    - 🚴 45 min cycling
    - 💪 20 min bodyweight exercises
    - 🏃 15 min jump rope
    """)
else:
    st.write("""
    **Intense Exercise Plan:**
    - 🔥 60 min HIIT training
    - 🏋️ 30 min weight lifting
    - 🏃‍♂️ 15 min treadmill running
    """)

# Workout History
if st.session_state.workouts:
    st.subheader("📋 Workout History")
    history_df = pd.DataFrame(st.session_state.workouts)
    st.dataframe(history_df)
    
    total_calories = sum(w['calories'] for w in st.session_state.workouts)
    avg_heart_rate = sum(w['heart_rate'] for w in st.session_state.workouts) / len(st.session_state.workouts)
    st.metric("Total Calories Burned", f"{total_calories:.0f} kcal")
    st.metric("Average Heart Rate", f"{avg_heart_rate:.0f} bpm")
    
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Workout History",
        data=csv,
        file_name="workout_history.csv",
        mime="text/csv"
    )

# About the Model
with st.expander("🔍 About the Prediction Model"):
    st.write("""
    The calorie burn prediction uses a Random Forest Regressor trained on workout data.
    It considers age, gender, BMI, duration, heart rate, and body temperature.
    """)
    importances = model.feature_importances_
    features = X.columns
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    st.bar_chart(feat_imp)

# Footer
st.markdown("---")
