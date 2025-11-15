"""Minimal Streamlit demo for LumaFin categorization API."""
import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="LumaFin Demo", page_icon="💳")
st.title("LumaFin – Transaction Categorization")

with st.form("txn_form"):
    merchant = st.text_input("Merchant", "Starbucks")
    amount = st.number_input("Amount", min_value=0.0, value=4.25, step=0.01)
    date = st.date_input("Date")
    description = st.text_input("Description", "morning coffee")
    user_id = st.number_input("User ID (optional)", min_value=0, value=0, step=1)
    submitted = st.form_submit_button("Categorize")

if submitted:
    payload = {
        "merchant": merchant,
        "amount": amount,
        "date": str(date),
        "description": description,
    }
    if user_id:
        payload["user_id"] = int(user_id)
    try:
        resp = requests.post(f"{API_URL}/categorize", json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Category: {data['category']['name']} (conf={round(data['category']['confidence']*100,1)}%)")
            st.caption(f"Decision path: {data['explanation']['decision_path']}")
            if data['explanation'].get('nearest_examples'):
                st.subheader("Nearest examples")
                st.table(data['explanation']['nearest_examples'])
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
