
from db import charts_collection
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import base64
import uuid


from generate_chart import ChartGenerator

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
    print("REQUEST RECEIVED")

    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format"}

        chart_generator = ChartGenerator(df)

        generated_code = chart_generator.generate_charts_code(
            chart_generator.col_dt_list,
            query
        )

        request_id = str(uuid.uuid4())
        charts_dir = f"charts/{request_id}"

        chart_generator.execute_generated_code(generated_code, charts_dir)

        images = []
        chart_paths = []

        for img_file in os.listdir(charts_dir):
            if img_file.endswith(".png"):

                path = os.path.join(charts_dir, img_file)

                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                images.append({
                    "filename": img_file,
                    "image_base64": encoded
                })

                chart_paths.append({
                    "filename": img_file,
                    "path": path
                })
        # SAVE TO MONGODB HERE
        charts_collection.insert_one({
            "request_id": request_id,
            "query": query,
            "file_name": file.filename,
            "charts": chart_paths,
            "created_at": datetime.now(timezone.utc)
        })

        return {
            "request_id": request_id,
            "charts": images
        }

    except Exception as e:
        return {
            "error": str(e)
        }