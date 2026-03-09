from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import io
from generate_chart import ChartGenerator

app = FastAPI()


# Health check
@app.get("/health")
def health_check():
    return {"status": "successful"}


@app.post("/generate-code")
async def generate_code(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    contents = await file.read()

    # Read file into dataframe
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return {"error": "Unsupported file format"}

    chart_generator = ChartGenerator(df)

    # generate LLM code only
    generated_code = chart_generator.generate_charts_code(
        chart_generator.col_dt_list,
        query
    )

    return {
        "generated_code": generated_code
    }