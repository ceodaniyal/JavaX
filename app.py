import streamlit as st
from generate_chart import ChartGenerator
import pandas as pd

st.title("Auto Chart Generator")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
user_query = st.text_area("Enter your chart requirement")

if st.button("Generate Charts"):

    if uploaded_file is None:
        st.error("Please upload a file.")
    else:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        chart_generator = ChartGenerator(df)

        if user_query.strip() == "":
            user_query = "Generate relevant charts."

        with st.spinner("Generating charts..."):
            chart_generator.run_ifelsequery(user_query)

        st.success("Charts generated successfully! Check 'charts' folder.")