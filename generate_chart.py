import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class ChartGenerator:
    def __init__(self, data):
        self.data = data
        self.col_dt_list = [(col, dt) for col, dt in zip(data.columns, data.dtypes)]

    @staticmethod
    def read_data(file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    def generate_charts_code(self, col_dt_list, query):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an Agentic AI Data Analyst and Visualization Engine.

Your task is to analyze provided dataset column names and their data types
and generate Python code for meaningful data visualization charts.

STRICT RULES:
- Output ONLY executable Python code.
- Do NOT include explanations, markdown formatting, or extra text.
- Assume a pandas DataFrame named 'df' already exists.
- Import: os, matplotlib.pyplot as plt, seaborn as sns.
- Include: os.makedirs('charts', exist_ok=True)
- Save every chart using:
  plt.savefig(f'charts/{filename}.png')
- Call plt.close() after saving each chart.
- Do NOT use display(), show(), or print().

ANALYSIS RULES:
- Detect numerical, categorical, and datetime columns automatically.
- For numerical columns → histograms, boxplots, correlation heatmaps.
- For categorical columns → bar charts.
- For datetime columns → trend/time-series plots.
- Avoid redundant or meaningless charts.

OUTPUT FORMAT RULE:
- The response must contain ONLY Python code.
- Each chart must be separated using triple double quotes.

Example format:

### chart 1
chart code
plt.savefig(...)

### chart 2
chart code
plt.savefig(...)

### chart 3
chart code
plt.savefig(...)
""",
                },
                {
                    "role": "user",
                    "content": f"Here are the column names and their data types: {col_dt_list}\n{query}",
                },
            ],
            extra_body={"reasoning": {"enabled": True}},
        )

        return response.choices[0].message.content.strip()