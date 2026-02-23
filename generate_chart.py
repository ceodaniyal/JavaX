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

    def generate_charts_code(self, col_dt_list):
        client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=os.getenv('OPENROUTER_API_KEY')
        )

        response = client.chat.completions.create(
        model="arcee-ai/trinity-large-preview:free",
        messages=[
                # {
                #     "role": "system",
                #     "content": "Based on the provided column names and data types, generate a Python code snippet that creates Analysis Charts using maybe matplotlib or seaborn. The charts should be relevant to the data types of the columns. For example, if there are numerical columns, create histograms or scatter plots. If there are categorical columns, create bar charts or pie charts. Ensure that the code is well-commented and easy to understand."
                # },
                    {
                    "role": "system",
                    "content": """
                    Based on the provided column names and data types, generate ONLY executable Python code. 
                    Do not add explanations. 
                    The code must:
                    1. Assume a pandas DataFrame named 'df' already exists.
                    2. Import 'os', 'matplotlib.pyplot as plt', and 'seaborn as sns'.
                    3. Include a line to create a directory named 'charts' if it doesn't exist: os.makedirs('charts', exist_ok=True).
                    4. Save each chart into the 'charts' folder using plt.savefig(f'charts/{filename}.png').
                    5. Use plt.close() after saving.
                    6. Do NOT use display() or show().
                    """
                    },
                    {
                        "role": "user",
                        "content": f"Here are the column names and their data types: {col_dt_list}"
                    }
                ],
        extra_body={"reasoning": {"enabled": True}}
        )

        # Extract the assistant message with reasoning_details
        response = response.choices[0].message

        # print("Assistant's response:", response.content)
        return response.content
    
    def execute_generated_code(self, generated_code):
        # Remove markdown if present
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()

        print("Generated Code:\n", generated_code)

        # Provide df inside execution context
        safe_globals = {
            "df": self.data,
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn"),
            "pd": __import__("pandas")
        }

        exec(generated_code, safe_globals)

    def run (self):
        generated_code = self.generate_charts_code(self.col_dt_list)
        self.execute_generated_code(generated_code)