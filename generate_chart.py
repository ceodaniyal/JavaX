import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
load_dotenv()


class ChartGenerator:
    def __init__(self, data):
        self.data = data
        self.col_dt_list = [(col, dt) for col, dt in zip(data.columns, data.dtypes)]

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
You are an AI data visualization generator.

Return ONLY Python code that generates charts.

Rules:
- DataFrame name is df
- Import matplotlib.pyplot as plt and seaborn as sns
Save every chart using:
plt.savefig(os.path.join(charts_dir, f"{filename}.png"))

A variable named charts_dir already exists and represents the folder where charts must be saved.
- Call plt.close() after each chart
- Do not print or show charts
"""
                },
                {
                    "role": "user",
                    "content": f"Columns and types: {col_dt_list}\nQuery: {query}"
                }
            ]
        )

        return response.choices[0].message.content.strip()

    def execute_generated_code(self, generated_code, charts_dir):

        os.makedirs(charts_dir, exist_ok=True)

        generated_code = generated_code.replace("```python", "").replace("```", "").strip()

        safe_globals = {
            "df": self.data,
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn"),
            "pd": __import__("pandas"),
            "os": __import__("os"),
            "charts_dir": charts_dir
        }

        try:
            exec(generated_code, safe_globals)
        except Exception as e:
            raise RuntimeError(f"Chart execution failed: {str(e)}")