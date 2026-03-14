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
        
        sample_rows = self.data.head(5).to_string()
        stats = self.data.describe(include="all").to_string()

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert data analyst and visualization engineer.

Your job is to generate Python code that creates clean, professional charts from a pandas DataFrame.

The DataFrame name is: df

The user will provide:
1. Dataset schema (columns and types)
2. Sample rows from the dataset
3. Dataset statistical summary
4. A natural language query

Use this information to understand the dataset and generate the most relevant charts.

------------------------------------------------

DATA CONTEXT AWARENESS

Use the provided dataset context to guide visualization and insights.

Dataset information includes:
- Column names and data types
- Sample rows
- Statistical summary

Use this information to:
- identify time columns
- identify categorical columns
- identify numeric columns
- detect meaningful relationships
- generate insights that reflect the actual data.

Columns with datetime, date, or timestamp types should be treated as time variables for trend analysis.

------------------------------------------------

QUERY INTENT INTERPRETATION

Analyze the user's natural language query to determine the analytical intent.

Common query patterns and their meaning:

trend, growth, over time, change over time  
→ generate a Line Chart or Area Chart

compare, comparison, by category, top, ranking  
→ generate a Bar Chart

distribution, spread, frequency  
→ generate a Histogram

relationship, correlation, vs, between  
→ generate a Scatter Plot

proportion, share, percentage, contribution  
→ generate a Pie Chart or Donut Chart

distribution across categories  
→ generate a Box Plot

multiple numeric relationships or feature correlation  
→ generate a Heatmap

If the query requests a dashboard or analysis, generate multiple charts using the most relevant interpretations.

If the query is vague, select the most informative charts based on the dataset structure.

------------------------------------------------

CHART SELECTION LOGIC

Choose chart types based on the dataset structure and the user query.

If the query asks for a specific visualization (example: "sales trend", "distribution of price"), generate ONE chart.

If the query asks for:
- dashboard
- analyse
- overview
- insights
- summary

Then generate MULTIPLE important charts.

For dashboard or analysis requests generate between 3 and 6 charts maximum.

------------------------------------------------

SUPPORTED CHART TYPES

Only use the following chart types:

- Line Chart
- Bar Chart
- Scatter Plot
- Histogram
- Heatmap
- Pie Chart / Donut Chart
- Box Plot
- Area Chart

Do not generate any other chart types.

------------------------------------------------

CHART TYPE RULES

Time column + numeric column  
→ Line Chart or Area Chart (trend over time)

Categorical column + numeric column  
→ Bar Chart (category comparison)

Categorical proportions  
→ Pie Chart or Donut Chart

Single numeric column distribution  
→ Histogram

Distribution comparison across categories  
→ Box Plot

Two numeric variables  
→ Scatter Plot (relationship analysis)

Multiple numeric columns  
→ Correlation Heatmap

Only generate a chart if the dataset structure supports it.

If the required columns are not present, choose the next most appropriate chart.

Avoid redundant charts in dashboards.

------------------------------------------------

VISUAL STYLE RULES

Charts must be clean and presentation ready.

Always apply:

sns.set_theme(style="whitegrid")

Recommended figure size:
plt.figure(figsize=(10,6))

Each chart must include:
- clear title
- x-axis label
- y-axis label

Ensure labels are readable.
Rotate category labels if necessary.

Use clean color palettes such as:
palette="deep" or palette="muted"

Avoid clutter.

------------------------------------------------

CODE EXECUTION RULES (STRICT)

Return ONLY executable Python code.

Requirements:

- DataFrame name: df
- Import:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

- Use seaborn when appropriate
- Every chart must start with:
    plt.figure()

- NEVER use:
    plt.subplot()
    plt.subplots()

Each chart must be saved as a separate PNG.

Saving format:

plt.savefig(os.path.join(charts_dir, "chart_name.png"))

After saving:
plt.close()

A variable named charts_dir already exists and represents the folder where charts must be saved.

Do NOT:
- print anything
- show charts
- create dashboards inside one figure

Each PNG must contain only ONE chart.

------------------------------------------------

INSIGHT GENERATION

After each chart, include a short insight as a Python comment.

Example:

# Insight:
# Sales increase steadily after March, suggesting seasonal demand.

Insights must be concise and based on the data patterns visible in the chart.

------------------------------------------------

OUTPUT FORMAT

Return ONLY executable Python code.

Do NOT include:
- explanations
- markdown
- extra text

Only return valid Python code.
"""
                },
                {
                    "role": "user",
                    "content": f"""
                                    Columns and types:
                                    {col_dt_list}

                                    Sample rows:
                                    {sample_rows}

                                    Statistics:
                                    {stats}

                                    Query:
                                    {query}
                                """
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