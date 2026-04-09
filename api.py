# from db import charts_collection
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logging
import pandas as pd
import io
import json
from plotly.utils import PlotlyJSONEncoder

from generate_chart import ChartGenerator

# ---------- LOGGER ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- APP ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "successful"}


@app.post("/generate-code")
async def generate_code(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        logger.info("Request received")

        contents = await file.read()
        logger.info(f"File received: {file.filename}")

        # ---------- LOAD DATA ----------
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
            logger.info("CSV file loaded into DataFrame")

        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
            logger.info("Excel file loaded into DataFrame")

        else:
            logger.error("Unsupported file format")
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format"}
            )

        # ---------- GENERATE CHARTS ----------
        chart_generator = ChartGenerator(df)
        logger.info("ChartGenerator initialized")

        result = chart_generator.generate(query)  # returns {"charts": [...], "tables": [...]}
        logger.info(f"Charts generated: {len(result['charts'])}  |  Tables: {len(result['tables'])}")

        # ---------- SERIALIZE ----------
        response_content = json.loads(
            json.dumps(result, cls=PlotlyJSONEncoder)
        )

        return JSONResponse(content=response_content)

    except Exception as e:
        logger.exception("Error in /generate-code endpoint")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )