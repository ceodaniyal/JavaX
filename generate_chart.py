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
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    def generate_charts_code(self, col_dt_list, query):
        client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=os.getenv('OPENROUTER_API_KEY')
        )

        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=[
                {
                    "role": "system",
                    "content": """
                You are an Agentic AI Data Analyst and Visualization Engine.

                Your task is to intelligently analyze the provided column names and data types and generate ONLY executable Python code for meaningful data analysis visualizations.

                Strict Rules:
                - Output ONLY valid executable Python code.
                - Do NOT include explanations, markdown, comments, or text outside code.
                - Assume a pandas DataFrame named 'df' already exists.
                - Import: os, matplotlib.pyplot as plt, seaborn as sns.
                - Include: os.makedirs('charts', exist_ok=True)
                - Save every chart inside the 'charts' folder using:
                plt.savefig(f'charts/{filename}.png')
                - Call plt.close() after every save.
                - Do NOT use display(), show(), or print().
                - In addition to charts, generate a pandas DataFrame named 'summary_df' that represents a meaningful pivot table or summary based on the query.
                - summary_df must remain available in the global scope.

                Analytical Behavior:
                - Automatically detect numerical, categorical, and datetime columns.
                - For numerical columns → create histograms, boxplots, or correlation heatmaps.
                - For categorical columns → create bar charts (top categories if high cardinality).
                - For datetime columns → create time trend plots if applicable.
                - Avoid redundant or meaningless charts.
                - Generate clean, readable, production-style plotting code.

                Be selective. Create only relevant and insightful charts.
                """
                },
                {
                    "role": "user",
                    "content": f"Here are the column names and their data types: {col_dt_list}\n{query}"
                }
            ],
            extra_body={"reasoning": {"enabled": True}}
        )

        response = response.choices[0].message
        return response.content
    
    def execute_generated_code(self, generated_code):
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()

        print("Generated Code:\n", generated_code)

        safe_globals = {
            "df": self.data,
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn"),
            "pd": __import__("pandas"),
            "os": __import__("os")
        }

        try:
            exec(generated_code, safe_globals)
        except Exception as e:
            print("Error executing generated code:", e)
            return None

        summary_df = safe_globals.get("summary_df", None)

        # Validate summary_df
        if summary_df is not None and isinstance(summary_df, pd.DataFrame):
            return summary_df
        else:
            print("Warning: summary_df not created by model.")
            return None

    def run_ifelsequery(self, query):
        if query and query.strip():
            generated_code = self.generate_charts_code(self.col_dt_list, query)
        else:
            generated_code = self.generate_charts_code(
                self.col_dt_list,
                "Generate relevant charts and a meaningful summary table."
            )

        summary_df = self.execute_generated_code(generated_code)
        return summary_df