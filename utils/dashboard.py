import streamlit as st
import plotly.express as px

def churn_by_contract(df):
    fig = px.histogram(df, x="Contract", color="Churn")
    st.plotly_chart(fig)

def churn_by_payment(df):
    fig = px.histogram(df, x="PaymentMethod", color="Churn")
    st.plotly_chart(fig)
