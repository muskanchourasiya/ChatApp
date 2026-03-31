import streamlit as st
import requests
import time

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("System Health Dashboard")

refresh = st.button("Refresh")
if refresh:
    st.rerun()

health = requests.get(f"{BACKEND}/health").json()

col1, col2, col3, col4 = st.columns(4)

col1.metric("API Status", health["status"])

metrics = requests.get(f"{BACKEND}/metrics").json()

col2.metric("Total Requests", metrics["total_requests"])
col3.metric("Active Sessions", metrics["active_sessions"])
col4.metric("Errors", metrics["total_errors"])

st.divider()

st.subheader("Performance")
st.metric("Last Response Time (s)", metrics["last_response_time"])

st.subheader("Recent Logs")

logs = requests.get(f"{BACKEND}/logs").json()["logs"]

for log in logs[::-1]:
    st.text(log)