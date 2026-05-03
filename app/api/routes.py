# #app/api/routes.py
# from datetime import datetime, timezone
# import asyncio
# import io
# import json
# import logging
# import uuid
# from pydantic import BaseModel

# import pandas as pd
# from fastapi import APIRouter, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from plotly.utils import PlotlyJSONEncoder

# from app.services.chart_service import ChartGenerator
# from app.utils.task_manager import create_task, cancel_task, remove_task, is_cancelled
# from app.services.chat_service import ChatService
# from app.utils.data_store import data_store

# # MongoDB imports
# from app.db.mongodb import (
#     save_file,
#     save_analysis,
#     update_analysis_result,
#     append_chat,
#     get_history as db_get_history,
#     get_analysis_by_id,
#     get_analysis_by_dataset_id,
#     get_file,
#     delete_analysis
# )

# router = APIRouter()

# # ---------- LOGGER ----------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )
# logger = logging.getLogger(__name__)

# def _coerce_numeric_columns(df):
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="ignore")
#     return df
# def read_file(contents, filename):
#     if filename.endswith(".csv"):
#         for enc in ["utf-8", "cp1252", "latin-1"]:
#             try:
#                 return pd.read_csv(io.BytesIO(contents), encoding=enc)
#             except UnicodeDecodeError:
#                 continue
#         raise ValueError("Cannot decode CSV")

#     elif filename.endswith(".xlsx"):
#         return pd.read_excel(io.BytesIO(contents))

#     else:
#         raise ValueError("Unsupported file format")
# # ─────────────────────────────────────────────
# @router.get("/health")
# def health_check():
#     return {"status": "successful"}

# # ─────────────────────────────────────────────
# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     contents = await file.read()
#     filename = file.filename or ""

#     df = read_file(contents, filename)

#     df = _coerce_numeric_columns(df)   # ← ADD THIS LINE

#     dataset_id = data_store.save(df)
#     logger.info("Dataset uploaded: %s (%d rows, %d cols, numeric: %s)",
#                 dataset_id, *df.shape, df.select_dtypes(include="number").columns.tolist())
#     return {"dataset_id": dataset_id}

# # ─────────────────────────────────────────────
# @router.post("/start-analysis")
# async def start_analysis(
#     file: UploadFile = File(...),
#     query: str = Form(...),
#     dataset_id: str = Form(None)
# ):
#     """
#     Accepts file + query directly (frontend-compatible).
#     Kicks off a background task and immediately returns a task_id.
#     The client polls /status/{task_id} and may call /cancel/{task_id}.
#     """
#     task_id = str(uuid.uuid4())
#     filename = file.filename or ""
#     logger.info("Starting analysis task %s | file=%s | query=%r | dataset_id=%s", task_id, filename, query[:60], dataset_id)

#     # ── parse uploaded file ──────────────────────────────────────────
#     contents = await file.read()
#     df = read_file(contents, filename)

#     # Save to data_store for memory cache (reuses ID if provided)
#     assigned_dataset_id = data_store.save(df, dataset_id=dataset_id)
    
#     # Save file to GridFS
#     file_id = save_file(contents, filename)
    
#     # Save analysis record to MongoDB
#     save_analysis(task_id, query, filename, assigned_dataset_id, file_id)

#     async def _run():
#         try:
#             generator = ChartGenerator(df)
#             result = await generator.generate(query, task_id)

#             if is_cancelled(task_id):
#                 update_analysis_result(task_id, "cancelled")
#                 logger.info("Task %s: cancelled after LLM — discarding result", task_id)
#                 return

#             serialised = json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
#             update_analysis_result(task_id, "completed", serialised)
#             logger.info("Task %s completed successfully", task_id)

#         except asyncio.CancelledError:
#             update_analysis_result(task_id, "cancelled")
#             logger.info("Task %s pipeline aborted via cancel flag", task_id)
#             raise

#         except Exception as exc:
#             update_analysis_result(task_id, "error", error=str(exc))
#             logger.exception("Task %s raised an exception", task_id)


#     create_task(task_id, _run())
#     return {"task_id": task_id}

# # ─────────────────────────────────────────────
# class ExistingAnalysisRequest(BaseModel):
#     dataset_id: str
#     query: str

# @router.post("/start-analysis-existing")
# async def start_analysis_existing(req: ExistingAnalysisRequest):
#     """
#     Generates new charts/analysis for an already uploaded dataset.
#     """
#     task_id = str(uuid.uuid4())
#     logger.info("Starting analysis task %s for existing dataset %s | query=%r", task_id, req.dataset_id, req.query[:60])

#     # 1) Get from memory
#     df = data_store.get(req.dataset_id)

#     # 2) Fallback to DB if not in memory
#     filename = ""
#     file_id = None
#     if df is None:
#         doc = get_analysis_by_dataset_id(req.dataset_id)
#         if doc and doc.get("file_id"):
#             file_id = doc["file_id"]
#             filename = doc.get("filename", "")
#             contents = get_file(file_id)
#             if contents:
#                 if filename.endswith(".csv"):
#                     df = pd.read_csv(io.BytesIO(contents))
#                 elif filename.endswith(".xlsx"):
#                     df = pd.read_excel(io.BytesIO(contents))
#                 if df is not None:
#                     data_store.save(df, dataset_id=req.dataset_id)
#     else:
#         # If it was in memory, still try to fetch filename/file_id for the record
#         doc = get_analysis_by_dataset_id(req.dataset_id)
#         if doc:
#             filename = doc.get("filename", "")
#             file_id = doc.get("file_id")

#     if df is None:
#         return JSONResponse(status_code=400, content={"error": "Invalid dataset_id or dataset expired"})

#     # Save analysis record to MongoDB
#     save_analysis(task_id, req.query, filename, req.dataset_id, file_id)

#     async def _run():
#         try:
#             generator = ChartGenerator(df)
#             result = await generator.generate(req.query, task_id)

#             if is_cancelled(task_id):
#                 update_analysis_result(task_id, "cancelled")
#                 logger.info("Task %s: cancelled after LLM — discarding result", task_id)
#                 return

#             serialised = json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
#             update_analysis_result(task_id, "completed", serialised)
#             logger.info("Task %s completed successfully", task_id)

#         except asyncio.CancelledError:
#             update_analysis_result(task_id, "cancelled")
#             logger.info("Task %s pipeline aborted via cancel flag", task_id)
#             raise

#         except Exception as exc:
#             update_analysis_result(task_id, "error", error=str(exc))
#             logger.exception("Task %s raised an exception", task_id)

#         finally:
#             remove_task(task_id)

#     create_task(task_id, _run())
#     return {"task_id": task_id}

# # ─────────────────────────────────────────────
# @router.get("/get-file/{dataset_id}")
# async def get_dataset_file(dataset_id: str):
#     """
#     Returns the original uploaded file for a dataset so the frontend
#     can re-use it for follow-up chart generation.
#     """
#     from fastapi.responses import Response
#     doc = get_analysis_by_dataset_id(dataset_id)
#     if not doc or not doc.get("file_id"):
#         return JSONResponse(status_code=404, content={"error": "File not found for this dataset"})

#     contents = get_file(doc["file_id"])
#     if contents is None:
#         return JSONResponse(status_code=404, content={"error": "File data not found in storage"})

#     filename = doc.get("filename", "data.csv")
#     content_type = "text/csv" if filename.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

#     return Response(
#         content=contents,
#         media_type=content_type,
#         headers={"Content-Disposition": f'attachment; filename="{filename}"'}
#     )

# # ─────────────────────────────────────────────
# @router.get("/status/{task_id}")
# async def get_status(task_id: str):
#     result = get_analysis_by_id(task_id)

#     if result:
#         return JSONResponse(content=result)

#     # If task exists but DB not updated yet → it's still running
#     return {"status": "running"}

# # ─────────────────────────────────────────────
# @router.post("/cancel/{task_id}")
# async def cancel(task_id: str):
#     """
#     Requests cancellation of a running task.
#     Returns {"cancelled": true/false}. Never 404s — frontend may send stale IDs.
#     """
#     result = get_analysis_by_id(task_id)
#     if result is None:
#         return {"cancelled": False}

#     cancelled = cancel_task(task_id)
#     return {"cancelled": cancelled}

# # ─────────────────────────────────────────────
# @router.delete("/history/{task_id}")
# async def delete_history_item(task_id: str):
#     """
#     Deletes an analysis task and its associated files.
#     """
#     success = delete_analysis(task_id)
#     if success:
#         return {"deleted": True}
#     return JSONResponse(status_code=404, content={"error": "Not found or could not be deleted"})

# # ─────────────────────────────────────────────
# @router.get("/history")
# async def get_history():
#     """
#     Returns a list of all tasks (newest first) for the sidebar.
#     Only returns metadata — not the full result data.
#     """
#     items = db_get_history()
#     return items

# # ─────────────────────────────────────────────
# @router.get("/history/{task_id}")
# async def get_history_item(task_id: str):
#     """
#     Returns full result data for a specific task (used when clicking sidebar item).
#     """
#     result = get_analysis_by_id(task_id)
#     if result is None:
#         return JSONResponse(status_code=404, content={"error": "Not found"})
#     return JSONResponse(content=result)

# # ─────────────────────────────────────────────
# @router.get("/chats/{dataset_id}")
# async def get_chats(dataset_id: str):
#     """
#     Returns the saved chat history for a dataset so the frontend can
#     restore previous conversations when the chat panel is re-opened.
#     """
#     doc = get_analysis_by_dataset_id(dataset_id)
#     if doc is None:
#         return JSONResponse(status_code=404, content={"error": "Dataset not found"})
#     chats = doc.get("chats", [])
#     # Convert timestamps to strings for JSON serialisation
#     from app.db.mongodb import _convert_datetimes
#     chats = _convert_datetimes(chats)
#     return {"chats": chats}

# # ─────────────────────────────────────────────
# class ChatRequest(BaseModel):
#     dataset_id: str
#     query: str
#     history: list[dict] = []

# @router.post("/chat")
# async def chat(req: ChatRequest):
#     df = data_store.get(req.dataset_id)

#     # Rehydrate dataframe from DB if missing in memory
#     if df is None:
#         doc = get_analysis_by_dataset_id(req.dataset_id)
#         if doc and doc.get("file_id"):
#             contents = get_file(doc["file_id"])
#             if contents:
#                 filename = doc.get("filename", "")
#                 df = read_file(contents, filename)
#                 if df is not None:
#                     data_store._store[req.dataset_id] = df

#     if df is None:
#         return JSONResponse(status_code=400, content={"error": "Invalid dataset_id or dataset expired"})

#     service = ChatService(df)
#     result = await service.chat(req.query, history=req.history)
    
#     # Save the chat to database
#     append_chat(req.dataset_id, req.query, result.get("answer"), result.get("table"))
    
#     return result

from datetime import datetime, timezone
import asyncio
import io
import json
import logging
import uuid
import traceback
import re as _re
from pydantic import BaseModel

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from plotly.utils import PlotlyJSONEncoder

from app.services.chart_service import ChartGenerator
from app.utils.task_manager import create_task, cancel_task, remove_task, is_cancelled
from app.services.chat_service import ChatService
from app.utils.data_store import data_store


router = APIRouter()

# ---------- LOGGER ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- IN-MEMORY RESULT STORE ----------
# Keyed by task_id → {"status": "running"|"completed"|"cancelled"|"error", ...}
_results: dict[str, dict] = {}

def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and convert string columns that are actually numeric values
    formatted with currency symbols, commas, or percent signs.

    Examples handled:
        ₹1,099  →  1099.0
        64%     →  64
        4.2     →  4.2  (already numeric — no-op)
        24,269  →  24269.0

    Safety gate: only converts a column if ≥60% of its non-null values
    parse successfully, preventing accidental conversion of free-text.
    """
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].astype(str).str.replace(
            r"[₹$€£¥,\s%]", "", regex=True
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        non_null = df[col].notna().sum()
        if non_null > 0 and parsed.notna().sum() / non_null >= 0.60:
            df[col] = parsed
            logger.info("Coerced column %r to numeric (dtype=%s)", col, parsed.dtype)
    return df

# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@router.get("/health")
def health_check():
    return {"status": "successful"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or ""

    try:
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

    except Exception as e:
        logger.exception("File parsing failed")
        return JSONResponse(status_code=500, content={"error": "Failed to parse file"})

    df = _coerce_numeric_columns(df)   # ← ADD THIS LINE

    dataset_id = data_store.save(df)
    logger.info("Dataset uploaded: %s (%d rows, %d cols, numeric: %s)",
                dataset_id, *df.shape, df.select_dtypes(include="number").columns.tolist())
    return {"dataset_id": dataset_id}

# ─────────────────────────────────────────────
# START ANALYSIS  (replaces the old /generate-code)
# ─────────────────────────────────────────────
@router.post("/start-analysis")
async def start_analysis(
    dataset_id: str = Form(...),
    query: str = Form(...),
):
    """
    Accepts dataset_id + query kicks off a background task, and immediately
    returns a task_id.  The client polls /status/{task_id} to track progress
    and may call /cancel/{task_id} at any time.
    """
    task_id = str(uuid.uuid4())
    logger.info("Starting analysis task %s for query: %r", task_id, query[:60])

    # # Read the file eagerly — the UploadFile object is not safe to pass into
    # # a background task because the request lifecycle may close it first.
    # contents = await file.read()
    # filename = file.filename or ""

    # if filename.endswith(".csv"):
    #     df = pd.read_csv(io.BytesIO(contents))
    # elif filename.endswith(".xlsx"):
    #     df = pd.read_excel(io.BytesIO(contents))
    # else:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": "Unsupported file format. Upload a .csv or .xlsx file."},
    #     )
    
    # dataset_id = data_store.save(df)

    df = data_store.get(dataset_id)

    if df is None:
        logger.warning("Invalid dataset_id: %s", dataset_id)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired dataset_id"},
        )

    

    # Mark as running immediately so the frontend sees a valid status on its
    # first poll even before the coroutine is scheduled.
    _results[task_id] = {"status": "running"}

    async def _run():
        try:
            generator = ChartGenerator(df)
            result = await generator.generate(query, task_id)

            # Check the flag one final time before writing 'completed' —
            # the LLM may have finished while a cancel was in-flight.
            if is_cancelled(task_id):
                _results[task_id] = {"status": "cancelled"}
                logger.info("Task %s: LLM finished but cancel was requested — discarding result", task_id)
                return

            serialised = json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
            _results[task_id] = {"status": "completed", "data": serialised}
            logger.info("Task %s completed successfully", task_id)

        except asyncio.CancelledError:
            # Raised by cooperative check inside the pipeline (not task.cancel()).
            _results[task_id] = {"status": "cancelled"}
            logger.info("Task %s pipeline aborted via cancel flag", task_id)
            raise

        except Exception as exc:
            _results[task_id] = {
                "status": "error",
                "error": str(exc),
                "trace": traceback.format_exc()
            }
            logger.exception("Task %s raised an exception", task_id)

        finally:
            remove_task(task_id)

    create_task(task_id, _run())

    return {"task_id": task_id }


# ─────────────────────────────────────────────
# STATUS POLL
# ─────────────────────────────────────────────
@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Returns the current state of the task.

    Possible responses
    ------------------
    {"status": "running"}
    {"status": "completed", "data": { ... }}
    {"status": "cancelled"}
    {"status": "error", "error": "..."}
    {"status": "not_found"}   ← unknown task_id
    """
    result = _results.get(task_id)
    if result is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# CANCEL
# ─────────────────────────────────────────────
@router.post("/cancel/{task_id}")
async def cancel(task_id: str):
    """
    Requests cancellation of a running task.
    If the task is already done the call is a no-op (cancelled: false).
    """
    if task_id not in _results:
        return JSONResponse(status_code=404, content={"error": "Unknown task_id"})

    cancelled = cancel_task(task_id)
    return {"cancelled": cancelled}

# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    dataset_id: str
    query: str
    history: list[dict] = []   # [{role: "user"|"assistant", content: str|dict}]

@router.post("/chat")
async def chat(req: ChatRequest):
    df = data_store.get(req.dataset_id)

    if df is None:
        return JSONResponse(status_code=400, content={"error": "Invalid dataset_id"})

    service = ChatService(df)
    result = await service.chat(req.query, history=req.history)

    return result