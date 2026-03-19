import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocess import preprocess_input
from utils.predict import load_models, make_prediction

st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    page_icon="📊"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned_churn_data.csv')

df = load_data()

# Fix churn column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Load models
rf_model, log_model = load_models()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Executive Summary", "Dashboard", "Prediction", "Business Impact"]
)

# =========================
# 📌 EXECUTIVE SUMMARY
# =========================
if page == "Executive Summary":
    st.title("📌 Customer Churn Intelligence")

    total = len(df)
    churn_rate = df['Churn'].mean()
    revenue = df['MonthlyCharges'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total)
    col2.metric("Churn Rate", f"{churn_rate:.2%}")
    col3.metric("Avg Monthly Revenue", f"${revenue:.2f}")

    st.markdown("---")

    st.subheader("📖 Key Insights")
    st.markdown("""
    - Month-to-month customers churn the most  
    - Electronic check users show high churn  
    - Low tenure customers are high-risk  
    - Long-term contracts reduce churn significantly  
    """)

    st.subheader("🎯 Recommendations")
    st.success("""
    Target:
    - New customers (<12 months)
    - Month-to-month contracts
    - Electronic check users  

    Strategy:
    - Offer discounts
    - Promote annual contracts
    """)

# =========================
# 📊 DASHBOARD
# =========================
elif page == "Dashboard":
    st.title("📊 Churn Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Contract", color="Churn",
                           title="Churn by Contract")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="PaymentMethod", color="Churn",
                           title="Churn by Payment Method")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tenure vs Churn")
    fig = px.box(df, x="Churn", y="tenure", color="Churn")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔮 PREDICTION
# =========================
elif page == "Prediction":
    st.title("🔮 Churn Prediction Tool")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure", 0, 72, 12)
        contract = st.selectbox("Contract",
                                ["Month-to-month", "One year", "Two year"])

    with col2:
        monthly = st.slider("Monthly Charges", 0, 150, 70)
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)",
                                "Credit card (automatic)"])

    input_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'Contract': contract,
        'PaymentMethod': payment
    }])

    processed = preprocess_input(input_df)

    model_choice = st.radio("Choose Model", ["Random Forest", "Logistic Regression"])

    if st.button("Predict"):
        model = rf_model if model_choice == "Random Forest" else log_model
        pred, prob = make_prediction(model, processed)

        if pred == 1:
            st.error(f"⚠️ High Churn Risk ({prob:.2%})")
        else:
            st.success(f"✅ Low Risk ({prob:.2%})")

# =========================
# 💰 ROI
# =========================
elif page == "Business Impact":
    st.title("💰 ROI Calculator")

    highrisk = df[
        (df['Contract'] == 'Month-to-month') &
        (df['tenure'] < 12) &
        (df['PaymentMethod'] == 'Electronic check')
    ]

    churn_rate = highrisk['Churn'].mean()
    total = len(highrisk)
    revenue = highrisk['MonthlyCharges'].mean()

    prevented = int(total * churn_rate * 0.25)
    savings = prevented * revenue
    cost = prevented * 10

    roi = (savings - cost) / cost if cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Target Customers", total)
    col2.metric("Prevented Churn", prevented)
    col3.metric("ROI", f"{roi:.2f}")
