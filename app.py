import os
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

            # Clear old charts
            if os.path.exists("charts"):
                for file in os.listdir("charts"):
                    if file.endswith(".png"):
                        os.remove(os.path.join("charts", file))

            # IMPORTANT: Capture returned summary_df
            summary_df = chart_generator.run_ifelsequery(user_query)

        st.success("Charts generated successfully!")

        # Display summary table if exists
        if summary_df is not None:
            st.subheader("Summary Table")
            st.dataframe(summary_df)

            csv = summary_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Summary CSV",
                data=csv,
                file_name="summary.csv",
                mime="text/csv",
            )
                # Display newly generated charts
        if os.path.exists("charts"):
            for file in os.listdir("charts"):
                if file.endswith(".png"):
                    st.image(os.path.join("charts", file), caption=file)