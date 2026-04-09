from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logging
import pandas as pd
import io
import json
from plotly.utils import PlotlyJSONEncoder

from app.services.chart_service import ChartGenerator

router = APIRouter()

# ---------- LOGGER ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

@router.get("/health")
def health_check():
    return {"status": "successful"}


@router.post("/generate-code")
async def generate_code(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        logger.info("Request received")

        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

        chart_generator = ChartGenerator(df)
        result = await chart_generator.generate(query)

        response_content = json.loads(
            json.dumps(result, cls=PlotlyJSONEncoder)
        )

        return JSONResponse(content=response_content)

    except Exception as e:
        logger.exception("Error in endpoint")
        return JSONResponse(status_code=500, content={"error": str(e)})